import os
import json
import re
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pathlib import Path

import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import stripe
from dotenv import load_dotenv

def load_environment(project_root=None):
    root = Path(project_root or Path(__file__).resolve().parent).resolve()
    load_dotenv(root / ".env", override=False)
    return root

PROJECT_ROOT = load_environment()

app = Flask(__name__)
# NEVER hardcode this in real code — set FLASK_SECRET_KEY in your environment.
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-key-change-me')

# 1. DATABASE SETUP
# Locally this defaults to SQLite (zero setup). In production, set DATABASE_URL
# to a Postgres connection string — most hosts (Render, Railway, etc.) have an
# ephemeral filesystem, so a SQLite file gets wiped on every deploy/restart.
_database_url = os.environ.get('DATABASE_URL', 'sqlite:///masomo.db')
if _database_url.startswith('postgres://'):
    # Some hosts still hand out the old 'postgres://' scheme; SQLAlchemy 1.4+/2.x needs 'postgresql://'.
    _database_url = _database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = _database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 2. LOGIN MANAGER SETUP
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'signup'  # new visitors are sent to sign up first; the page links to /login for existing users

# 3. STRIPE CONFIGURATION
# Your previous version had a live-looking secret key committed directly in
# source. Roll that key in the Stripe dashboard and load the new one from
# the environment instead — never commit it.
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')

STRIPE_PRICES = {
    "pro": os.environ.get('STRIPE_PRICE_PRO', ''),
    "premium": os.environ.get('STRIPE_PRICE_PREMIUM', ''),
    "maxpro": os.environ.get('STRIPE_PRICE_MAXPRO', ''),
}
YOUR_DOMAIN = os.environ.get('APP_DOMAIN', 'http://localhost:5000')

# -------------------------------------------------------------------------
# PASSWORD RESET (stateless tokens — no extra DB column needed)
# -------------------------------------------------------------------------
reset_serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
RESET_TOKEN_MAX_AGE = 3600  # 1 hour

# SMTP config — set these to actually send reset emails. If SMTP_HOST is
# unset, the reset link is written to the server log instead (handy for
# local dev), never shown in the browser.
SMTP_HOST = os.environ.get('SMTP_HOST')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
SMTP_USER = os.environ.get('SMTP_USER')
SMTP_PASS = os.environ.get('SMTP_PASS')
MAIL_FROM = os.environ.get('MAIL_FROM', SMTP_USER or 'no-reply@masomo.app')


def send_reset_email(to_email, reset_url):
    if not SMTP_HOST:
        app.logger.info(f"[password reset] SMTP not configured — reset link for {to_email}: {reset_url}")
        return

    msg = MIMEText(
        f"Someone requested a password reset for your Masomo account.\n\n"
        f"Reset your password here (link expires in 1 hour):\n{reset_url}\n\n"
        f"If you didn't request this, you can safely ignore this email."
    )
    msg['Subject'] = 'Reset your Masomo password'
    msg['From'] = MAIL_FROM
    msg['To'] = to_email

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        if SMTP_USER and SMTP_PASS:
            server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(MAIL_FROM, [to_email], msg.as_string())

# -------------------------------------------------------------------------
# GROQ CONFIGURATION (replaces Ollama)
# -------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')  # set this in your environment
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
MAX_MESSAGE_LENGTH = 800
MAX_HISTORY_MESSAGES = 12  # trim long conversations before sending to the API

if not GROQ_API_KEY:
    app.logger.warning("GROQ_API_KEY is not present in the process environment at startup")

# -------------------------------------------------------------------------
# DATABASE MODELS
# -------------------------------------------------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_premium = db.Column(db.Boolean, default=False)
    expiry_date = db.Column(db.DateTime, nullable=True)

    def set_password(self, raw_password):
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password):
        return check_password_hash(self.password_hash, raw_password)

    @property
    def subscription_expired(self):
        if self.is_premium:
            return False
        if not self.expiry_date:
            return True
        return datetime.now() > self.expiry_date


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------
def call_groq(messages):
    api_key = os.environ.get('GROQ_API_KEY') or GROQ_API_KEY
    if not api_key:
        app.logger.error("Groq request failed because GROQ_API_KEY is missing from the runtime environment")
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your server environment or to the project .env file."
        )
    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.3,
            },
            timeout=45,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Groq error: {str(e)}")
    except (KeyError, IndexError):
        raise RuntimeError("Groq returned an unexpected response shape")


def extract_json(text):
    """Strip markdown code fences and parse the JSON the model returned."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def describe_level(level, grade_form):
    if level == "cbc":
        return f"{grade_form or 'Grade 6'} (CBC) in Kenya"
    if level == "upper_primary":
        return f"{grade_form or 'Grade 4'} Upper Primary in Kenya"
    if level == "junior_school":
        return f"{grade_form or 'Grade 7'} Junior School in Kenya"
    if level == "senior_school":
        return f"{grade_form or 'Grade 10'} Senior School in Kenya"
    return f"{grade_form or 'Form 1'} secondary school (KCSE track) in Kenya"


def build_system_prompt(subject, level, grade_form=None):
    level_desc = describe_level(level, grade_form)
    return (
        f"You are a warm, patient, encouraging tutor helping a {level_desc} student with {subject}. "
        f"Explain concepts clearly, simply, and step by step. Use examples relevant to Kenya."
    )

# -------------------------------------------------------------------------
# AUTHENTICATION ROUTES (Login / Signup)
# -------------------------------------------------------------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Email and password are required.', 'danger')
            return redirect(url_for('signup'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'danger')
            return redirect(url_for('signup'))

        trial_end = datetime.now() + timedelta(days=1)
        new_user = User(email=email, is_premium=False, expiry_date=trial_end)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        return redirect(url_for('home'))

    return render_template('auth.html', mode='signup')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid credentials!', 'danger')

    return render_template('auth.html', mode='login')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = User.query.filter_by(email=email).first()

        if user:
            token = reset_serializer.dumps(email, salt='password-reset')
            reset_url = f"{YOUR_DOMAIN}/reset-password/{token}"
            try:
                send_reset_email(email, reset_url)
            except Exception as e:
                app.logger.error(f"Failed to send reset email to {email}: {e}")

        # Same message whether or not the email exists — avoids leaking
        # which addresses are registered.
        flash("If that email is registered, a reset link is on its way. Check your inbox.", "success")
        return redirect(url_for('login'))

    return render_template('auth.html', mode='forgot')


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = reset_serializer.loads(token, salt='password-reset', max_age=RESET_TOKEN_MAX_AGE)
    except SignatureExpired:
        flash("That reset link has expired. Please request a new one.", "danger")
        return redirect(url_for('forgot_password'))
    except BadSignature:
        flash("That reset link isn't valid. Please request a new one.", "danger")
        return redirect(url_for('forgot_password'))

    user = User.query.filter_by(email=email).first()
    if not user:
        flash("That reset link isn't valid. Please request a new one.", "danger")
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        if not password or len(password) < 8:
            flash("Password must be at least 8 characters.", "danger")
            return render_template('auth.html', mode='reset', token=token)

        if password != confirm:
            flash("Passwords don't match.", "danger")
            return render_template('auth.html', mode='reset', token=token)

        user.set_password(password)
        db.session.commit()
        flash("Password updated — you can log in now.", "success")
        return redirect(url_for('login'))

    return render_template('auth.html', mode='reset', token=token)

# -------------------------------------------------------------------------
# MAIN APP PAGE
# -------------------------------------------------------------------------
@app.route('/')
def home():
    if current_user.is_authenticated:
        return render_template(
            'index.html',
            user=current_user,
            subscription_expired=current_user.subscription_expired,
        )

    return render_template(
        'landing.html',
        title='Masomo — Study Help',
        description='A friendly AI study tutor for students in Kenya, with help for Mathematics, English, and more.',
    )


@app.route('/robots.txt')
def robots_txt():
    content = (
        "User-agent: *\n"
        f"Allow: /\n"
        f"Sitemap: {request.host_url}sitemap.xml\n"
    )
    return Response(content, mimetype='text/plain')


@app.route('/sitemap.xml')
def sitemap_xml():
    urls = [
        request.host_url,
        f"{request.host_url}login",
        f"{request.host_url}signup",
        f"{request.host_url}forgot-password",
    ]
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    for url in urls:
        xml += '  <url>\n'
        xml += f'    <loc>{url}</loc>\n'
        xml += '    <changefreq>weekly</changefreq>\n'
        xml += '    <priority>0.8</priority>\n'
        xml += '  </url>\n'
    xml += '</urlset>\n'
    return Response(xml, mimetype='application/xml')


@app.route('/logo.svg')
def serve_logo():
    return send_from_directory(PROJECT_ROOT, 'logo.svg', mimetype='image/svg+xml')

# -------------------------------------------------------------------------
# STRIPE PAYMENT ROUTES
# -------------------------------------------------------------------------
@app.route('/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    tier_selected = request.form.get('tier')  # expected: 'pro', 'premium', or 'maxpro'

    if tier_selected not in STRIPE_PRICES or not STRIPE_PRICES[tier_selected]:
        flash(
            f"The '{tier_selected}' plan isn't configured yet — its Stripe price ID is missing "
            f"from the server's environment variables. Double-check STRIPE_PRICE_"
            f"{(tier_selected or '').upper()} is set.",
            "danger",
        )
        return redirect(url_for('home'))

    if not stripe.api_key:
        flash("Payments aren't configured on this server yet — STRIPE_SECRET_KEY is missing.", "danger")
        return redirect(url_for('home'))

    chosen_price_id = STRIPE_PRICES[tier_selected]

    try:
        # Deliberately NOT passing payment_method_types here. Stripe's Checkout
        # dynamically shows whichever payment methods you've enabled in the
        # Dashboard (Settings > Payment methods) — including PayPal — to
        # eligible customers automatically. To turn PayPal on:
        #   1. Dashboard > Settings > Payment methods > PayPal > Enable
        #   2. Under PayPal's settings, also enable "Recurring payments"
        #      (required since these are subscriptions, not one-time charges;
        #      this can take up to 5 business days to activate, and is only
        #      available in supported business locations)
        checkout_session = stripe.checkout.Session.create(
            line_items=[{'price': chosen_price_id, 'quantity': 1}],
            mode='subscription',
            success_url=YOUR_DOMAIN + '/payment-success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=YOUR_DOMAIN + '/',
            client_reference_id=str(current_user.id),
        )
        return redirect(checkout_session.url, code=303)
    except Exception as e:
        flash(f"Payment error: {str(e)}", "danger")
        return redirect(url_for('home'))


@app.route('/payment-success')
@login_required
def payment_success():
    """
    NOTE: confirming payment here (client redirect) is only good enough for
    local testing. In production, mark the user premium from a verified
    Stripe webhook event (checkout.session.completed), not from this
    redirect, since a user can hit this URL directly without paying.
    """
    session_id = request.args.get('session_id')
    if session_id:
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            if session.payment_status == 'paid' and str(current_user.id) == session.client_reference_id:
                current_user.is_premium = True
                db.session.commit()
                flash('Payment successful — welcome to Masomo Premium!', 'success')
        except Exception as e:
            flash(f"Could not confirm payment: {str(e)}", "danger")
    return redirect(url_for('home'))


@app.route('/webhooks/stripe', methods=['POST'])
def stripe_webhook():
    """Verified server-to-server confirmation — the safe way to grant access."""
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    endpoint_secret = os.environ.get('STRIPE_WEBHOOK_SECRET')

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except (ValueError, stripe.error.SignatureVerificationError):
        return jsonify({"error": "invalid signature"}), 400

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        user_id = session.get('client_reference_id')
        if user_id:
            user = User.query.get(int(user_id))
            if user:
                user.is_premium = True
                db.session.commit()

    return jsonify({"received": True})

# -------------------------------------------------------------------------
# TUTOR API ROUTES (now backed by Groq instead of Ollama)
# -------------------------------------------------------------------------
@app.route('/api/tutor', methods=['POST'])
@login_required
def api_tutor():
    if current_user.subscription_expired:
        return jsonify({"error": "Your trial has ended — please choose a plan."}), 402

    data = request.get_json(silent=True) or {}
    subject = data.get('subject')
    level = data.get('level')
    grade_form = data.get('gradeForm')
    messages = data.get('messages', [])

    if not subject or not messages:
        return jsonify({"error": "subject and messages are required"}), 400

    system_prompt = build_system_prompt(subject, level, grade_form)
    groq_messages = [{"role": "system", "content": system_prompt}]
    for m in messages[-MAX_HISTORY_MESSAGES:]:
        role = 'assistant' if m.get('role') == 'assistant' else 'user'
        content = (m.get('content') or '')[:MAX_MESSAGE_LENGTH]
        groq_messages.append({"role": role, "content": content})

    try:
        answer = call_groq(groq_messages)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    return jsonify({"answer": answer})


@app.route('/api/topics', methods=['POST'])
@login_required
def api_topics():
    data = request.get_json(silent=True) or {}
    subject = data.get('subject')
    level = data.get('level')
    grade_form = data.get('gradeForm')

    if not subject:
        return jsonify({"error": "subject is required"}), 400

    level_desc = describe_level(level, grade_form)
    prompt = (
        f"List 6 to 9 study topics for {subject} for a {level_desc} student, "
        f'as a JSON array of short topic name strings, e.g. ["Topic 1", "Topic 2"]. '
        f"Return ONLY the JSON array — no commentary, no markdown fences."
    )

    try:
        raw = call_groq([{"role": "user", "content": prompt}])
        topics = extract_json(raw)
        if not isinstance(topics, list):
            raise ValueError("Expected a JSON array of topics")
    except (RuntimeError, ValueError, json.JSONDecodeError) as e:
        return jsonify({"error": f"Could not generate topics: {str(e)}"}), 502

    return jsonify({"topics": topics})


@app.route('/api/generate-test', methods=['POST'])
@login_required
def api_generate_test():
    data = request.get_json(silent=True) or {}
    subject = data.get('subject')
    level = data.get('level')
    grade_form = data.get('gradeForm')
    topic = data.get('topic')
    all_topics = data.get('allTopics', [])

    if not subject:
        return jsonify({"error": "subject is required"}), 400

    level_desc = describe_level(level, grade_form)
    scope = f"the topic '{topic}'" if topic else f"all of these topics: {', '.join(all_topics)}"

    prompt = f"""Create a short test for a {level_desc} student on {subject}, covering {scope}.
Return ONLY valid JSON in exactly this shape, no commentary, no markdown fences:
{{
  "total_marks": <int>,
  "questions": [
    {{"id": 1, "type": "mcq", "question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "correct_option": "A", "marks": 1, "explanation": "..."}},
    {{"id": 2, "type": "short", "question": "...", "marks": 3, "model_answer": "...", "marking_points": ["...", "...", "..."]}}
  ]
}}
Include 5 to 8 questions total, mixing "mcq" and "short" types. Number "id" sequentially starting at 1."""

    try:
        raw = call_groq([{"role": "user", "content": prompt}])
        test_data = extract_json(raw)
        if "questions" not in test_data or "total_marks" not in test_data:
            raise ValueError("Response missing required fields")
    except (RuntimeError, ValueError, json.JSONDecodeError) as e:
        return jsonify({"error": f"Could not generate test: {str(e)}"}), 502

    return jsonify(test_data)


@app.route('/healthz')
def healthz():
    """Simple health check most hosting platforms ping to confirm the app is alive."""
    return jsonify({"status": "ok"})


@app.route('/debug/config')
def debug_config():
    """
    TEMPORARY diagnostic route — shows which env vars actually loaded, without
    leaking full secret values. Delete this route once things are working;
    don't leave it in a production deploy.
    """
    def mask(value):
        if not value:
            return None
        return value[:6] + "..." + f"({len(value)} chars)"

    return jsonify({
        "STRIPE_SECRET_KEY": mask(os.environ.get('STRIPE_SECRET_KEY')),
        "STRIPE_PRICE_PRO": mask(os.environ.get('STRIPE_PRICE_PRO')),
        "STRIPE_PRICE_PREMIUM": mask(os.environ.get('STRIPE_PRICE_PREMIUM')),
        "STRIPE_PRICE_MAXPRO": mask(os.environ.get('STRIPE_PRICE_MAXPRO')),
        "GROQ_API_KEY": mask(os.environ.get('GROQ_API_KEY')),
        "DATABASE_URL": mask(os.environ.get('DATABASE_URL')),
        "APP_DOMAIN": os.environ.get('APP_DOMAIN'),
    })


# Create tables on import, not just when run directly — gunicorn imports this
# module as `app:app` and never executes the `if __name__ == '__main__'` block below.
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode)

