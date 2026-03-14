"""
AI Study Tracker - Flask Backend
All API endpoints for authentication, study sessions, AI predictions, and study plans.
"""

import os
import sys
import json
import pickle
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from database import init_db, get_db, create_notification

app = Flask(__name__, static_folder='../', static_url_path='')
app.secret_key = secrets.token_hex(32)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
# Allow all origins for local dev with credentials
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, X-User-Id, X-Username')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    return response


@app.route("/")
def home():
    return "AI Study Tracker Running"

# ── ML Model Paths ─────────────────────────────────────────────────────────────
ML_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml')
PRODUCTIVITY_MODEL_PATH = os.path.join(ML_DIR, 'productivity_model.pkl')
PLACEMENT_MODEL_PATH = os.path.join(ML_DIR, 'placement_model.pkl')
SCALER_PATH = os.path.join(ML_DIR, 'scaler.pkl')

productivity_model_data = None
placement_model_data = None
scaler = None

def load_models():
    global productivity_model_data, placement_model_data, scaler
    if os.path.exists(PRODUCTIVITY_MODEL_PATH):
        with open(PRODUCTIVITY_MODEL_PATH, 'rb') as f:
            productivity_model_data = pickle.load(f)
        print("[INFO] Productivity model loaded.")
    else:
        print("[WARN] Productivity model not found. Run ml/train_model.py first.")

    if os.path.exists(PLACEMENT_MODEL_PATH):
        with open(PLACEMENT_MODEL_PATH, 'rb') as f:
            placement_model_data = pickle.load(f)
        print("[INFO] Placement readiness model loaded.")
    else:
        print("[WARN] Placement model not found. Run ml/train_model.py first.")

    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("[INFO] Scaler loaded.")

# ── FAQ Data ──────────────────────────────────────────────────────────────────
FAQ_PATH = os.path.join(os.path.dirname(__file__), 'faq.json')
faq_data = []

def load_faq():
    global faq_data
    if os.path.exists(FAQ_PATH):
        try:
            with open(FAQ_PATH, 'r') as f:
                faq_data = json.load(f)
            print(f"[INFO] FAQ data loaded from {FAQ_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to load FAQ: {e}")
    else:
        print("[WARN] faq.json not found.")

from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")

if GENAI_API_KEY and GENAI_API_KEY != "YOUR_API_KEY_HERE":
    genai.configure(api_key=GENAI_API_KEY)
    print("[INFO] Gemini API configured successfully.")
else:
    print("[WARN] Gemini API key not found or placeholder used. Falling back to simulated responses.")

# ── LLM Logic ────────────────────────────────────────────────────────────────
def real_llm_query(query):
    """
    Attempts to query the Gemini API for questions not in the FAQ.
    Falls back to a smarter heuristic model if API fails or key is missing.
    """
    if not GENAI_API_KEY or GENAI_API_KEY == "YOUR_API_KEY_HERE":
        return simulated_llm_query(query)

def real_llm_query(query):
    """
    Queries Gemini API. Returns (answer_text, source_label).
    """
    if not GENAI_API_KEY or GENAI_API_KEY == "YOUR_API_KEY_HERE":
        return simulated_llm_query(query), "AI Engine (Simulated)"

    # List of models to try in order
    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest"]
    
    for model_name in models_to_try:
        try:
            print(f"[DEBUG] Attempting Gemini with model: {model_name}")
            model = genai.GenerativeModel(model_name)
            
            system_context = (
                "You are the AI Study Assistant. Answer the following student query professionally and concisely. "
                "The users want high-quality answers like ChatGPT or Claude. "
                "Focus on education and study tips. "
                "User Query: "
            )
            
            response = model.generate_content(system_context + query)
            
            if response and response.text:
                print(f"[DEBUG] Gemini success with {model_name}")
                return response.text.strip(), "Gemini AI"
            
        except Exception as e:
            print(f"[ERROR] Gemini failed for {model_name}: {e}")
            continue # Try next model
            
    # If all models fail, return simulated
    print("[WARN] All Gemini models failed or quota exceeded. Falling back to simulated.")
    return simulated_llm_query(query), "AI Engine (Simulated)"

def simulated_llm_query(query):
    """
    Smarter fallback responses for common queries when API is hit.
    """
    query_lower = query.lower()
    
    # Common Technical Questions
    if 'html' in query_lower and 'full form' in query_lower:
        return "HTML stands for **HyperText Markup Language**. It is the standard language used to create the structure of web pages."
    elif 'css' in query_lower and 'full form' in query_lower:
        return "CSS stands for **Cascading Style Sheets**. It defines the layout and styling (colors, fonts, etc.) of a web document."
    
    # Study Plans
    if 'python' in query_lower and ('plan' in query_lower or 'roadmap' in query_lower):
        return (
            "Here is a 4-week **Python Study Roadmap**:\n\n"
            "**Week 1: Basics** (Syntax, Variables, Data Types, Input/Output)\n"
            "**Week 2: Control Flow** (Loops, If-Else, Functions, Modules)\n"
            "**Week 3: Data Structures** (Lists, Tuples, Dictionaries, Sets, List Comprehensions)\n"
            "**Week 4: Advanced Concepts** (File Handling, Exception Handling, Intro to OOP)\n\n"
            "Consistent practice on platforms like LeetCode or HackerRank is highly recommended!"
        )
    
    # Generic AI Advice
    if 'why' in query_lower:
        return "Success in your studies often depends on two core pillars: **Deep Work** (focused study without distraction) and **Active Recall** (testing yourself frequently). Tracking your focus in the dashboard can help you find your most productive hours."
    elif 'tips' in query_lower or 'improve' in query_lower:
        return "To improve your productivity, try the **Pomodoro Technique**: study for 25 minutes, then take a 5-minute break. This prevents burnout and keeps your brain fresh. Also, ensure you have a dedicated study space."
    elif any(greet in query_lower for greet in ['hello', 'hi', 'hey']):
        return "Hello! I'm your AI Study Assistant. I can help with study advice, technical definitions, or placement tips. What can I help you with today?"
    elif 'dsa' in query_lower or 'algorithms' in query_lower:
        return "For DSA, start by mastering **Arrays and Strings**. Then move to **Linked Lists, Stacks, and Queues**. The key is to understand the complexity (Time & Space) of each algorithm, not just the code."
    else:
        return (
            f"It looks like you're asking about '{query}'. While I'm in simulated mode (due to API limits), here's some advice: "
            "Break your goal into small chunks, track your progress daily, and prioritize sleep for better memory retention. "
            "Would you like a study plan for a specific subject?"
        )

# ── Helpers ────────────────────────────────────────────────────────────────────
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized', 'logged_in': False}), 401
        return f(*args, **kwargs)
    return decorated

def get_current_user():
    if 'user_id' not in session:
        return None
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    db.close()
    return user

# ── Auth Routes ────────────────────────────────────────────────────────────────
@app.before_request
def load_user_from_header():
    uid = request.headers.get('X-User-Id')
    if uid:
        session['user_id'] = int(uid)
        username = request.headers.get('X-Username')
        if username:
            session['username'] = username

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')

    if not username or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    db = get_db()
    existing = db.execute('SELECT id FROM users WHERE username=? OR email=?', (username, email)).fetchone()
    if existing:
        db.close()
        return jsonify({'error': 'Username or email already exists'}), 409

    hashed = hash_password(password)
    db.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
               (username, email, hashed))
    db.commit()
    user = db.execute('SELECT id FROM users WHERE username=?', (username,)).fetchone()
    db.close()

    session['user_id'] = user['id']
    session['username'] = username
    create_notification(user['id'], f"Welcome to AI Study Tracker, {username}! 🎓", 'success')
    return jsonify({'message': 'Registration successful', 'username': username, 'user_id': user['id']}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    hashed = hash_password(password)
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE (username=? OR email=?) AND password_hash=?',
                      (username, username, hashed)).fetchone()
    db.close()

    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401

    session['user_id'] = user['id']
    session['username'] = user['username']
    return jsonify({'message': 'Login successful', 'username': user['username'], 'user_id': user['id']}), 200

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/check_auth', methods=['GET'])
def check_auth():
    if 'user_id' in session:
        return jsonify({'logged_in': True, 'username': session.get('username'), 'user_id': session['user_id']}), 200
    return jsonify({'logged_in': False}), 200

# ── Study Session Routes ───────────────────────────────────────────────────────
@app.route('/add_session', methods=['POST'])
@login_required
def add_session():
    data = request.get_json()
    subject = data.get('subject', '').strip()
    start_time = data.get('start_time', '')
    end_time = data.get('end_time', '')
    duration_hours = float(data.get('duration_hours', 0))
    self_rating = int(data.get('self_rating', 3))
    focus_level = int(data.get('focus_level', 5))
    date = data.get('date', datetime.now().strftime('%Y-%m-%d'))

    if not subject or not start_time or not end_time:
        return jsonify({'error': 'Subject, start time, and end time are required'}), 400
    if duration_hours <= 0:
        return jsonify({'error': 'Duration must be positive'}), 400

    db = get_db()
    db.execute('''INSERT INTO study_sessions
                  (user_id, subject, start_time, end_time, duration_hours, self_rating, focus_level, date)
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
               (session['user_id'], subject, start_time, end_time, duration_hours, self_rating, focus_level, date))
    db.commit()
    db.close()

    # Check for notifications
    _check_and_create_notifications(session['user_id'])
    return jsonify({'message': f'Study session for {subject} added successfully!'}), 201

@app.route('/delete_session/<int:session_id>', methods=['DELETE'])
@login_required
def delete_session(session_id):
    uid = session['user_id']
    db = get_db()
    db.execute('DELETE FROM study_sessions WHERE id=? AND user_id=?', (session_id, uid))
    db.commit()
    db.close()
    return jsonify({'message': 'Session deleted'}), 200

# ── Collaborative Study Routes ──────────────────────────────────────────────────
@app.route('/create_group', methods=['POST'])
@login_required
def create_group():
    data = request.get_json()
    name = data.get('name', '').strip()
    desc = data.get('description', '').strip()
    if not name:
        return jsonify({'error': 'Group name is required'}), 400
    
    db = get_db()
    cursor = db.execute('INSERT INTO study_groups (name, description, created_by) VALUES (?, ?, ?)',
                       (name, desc, session['user_id']))
    group_id = cursor.lastrowid
    # Add creator as admin member
    db.execute('INSERT INTO group_members (group_id, user_id, role) VALUES (?, ?, ?)',
               (group_id, session['user_id'], 'admin'))
    db.commit()
    db.close()
    return jsonify({'message': 'Group created!', 'group_id': group_id}), 201

@app.route('/get_my_groups', methods=['GET'])
@login_required
def get_my_groups():
    db = get_db()
    groups = db.execute('''
        SELECT g.*, (SELECT COUNT(*) FROM group_members WHERE group_id = g.id) as member_count
        FROM study_groups g
        JOIN group_members m ON g.id = m.group_id
        WHERE m.user_id = ?
    ''', (session['user_id'],)).fetchall()
    db.close()
    return jsonify([dict(g) for g in groups]), 200

@app.route('/join_group', methods=['POST'])
@login_required
def join_group():
    data = request.get_json()
    group_id = data.get('group_id')
    if not group_id:
        return jsonify({'error': 'Group ID required'}), 400
    
    db = get_db()
    try:
        db.execute('INSERT INTO group_members (group_id, user_id) VALUES (?, ?)',
                   (group_id, session['user_id']))
        db.commit()
        return jsonify({'message': 'Joined group successfully!'}), 200
    except:
        return jsonify({'error': 'Already a member or group not found'}), 400
    finally:
        db.close()

@app.route('/get_group_details/<int:group_id>', methods=['GET'])
@login_required
def get_group_details(group_id):
    db = get_db()
    group = db.execute('SELECT * FROM study_groups WHERE id = ?', (group_id,)).fetchone()
    if not group:
        db.close()
        return jsonify({'error': 'Group not found'}), 404
    
    members = db.execute('''
        SELECT u.username, m.role, m.joined_at, 
                (SELECT (strftime('%s', 'now') - strftime('%s', start_time)) / 60 FROM active_sessions WHERE user_id = u.id) as active_minutes
        FROM users u
        JOIN group_members m ON u.id = m.user_id
        WHERE m.group_id = ?
    ''', (group_id,)).fetchall()
    
    # Simple group stats
    stats = db.execute('''
        SELECT SUM(duration_hours) as total_hours, AVG(focus_level) as avg_focus
        FROM study_sessions
        WHERE user_id IN (SELECT user_id FROM group_members WHERE group_id = ?)
    ''', (group_id,)).fetchone()
    
    db.close()
    return jsonify({
        'group': dict(group),
        'members': [dict(m) for m in members],
        'stats': dict(stats) if stats else {'total_hours': 0, 'avg_focus': 0}
    }), 200

@app.route('/group_chat/<int:group_id>', methods=['GET'])
@login_required
def get_group_chat(group_id):
    db = get_db()
    # verify membership
    member = db.execute('SELECT 1 FROM group_members WHERE group_id=? AND user_id=?',
                       (group_id, session['user_id'])).fetchone()
    if not member:
        db.close()
        return jsonify({'error': 'Forbidden'}), 403
        
    msgs = db.execute('''
        SELECT m.*, u.username 
        FROM group_messages m
        JOIN users u ON m.user_id = u.id
        WHERE m.group_id = ?
        ORDER BY m.created_at ASC LIMIT 50
    ''', (group_id,)).fetchall()
    db.close()
    return jsonify([dict(m) for m in msgs]), 200

@app.route('/send_group_message', methods=['POST'])
@login_required
def send_group_message():
    data = request.get_json()
    group_id = data.get('group_id')
    msg = data.get('message', '').strip()
    if not group_id or not msg:
        return jsonify({'error': 'Missing fields'}), 400
        
    db = get_db()
    db.execute('INSERT INTO group_messages (group_id, user_id, message) VALUES (?, ?, ?)',
               (group_id, session['user_id'], msg))
    db.commit()
    db.close()
    return jsonify({'message': 'Sent'}), 201

@app.route('/group_leaderboard/<int:group_id>', methods=['GET'])
@login_required
def get_group_leaderboard(group_id):
    db = get_db()
    # Study Points = (Study Hours × 10) + Focus Score + Study Streak Bonus (simulated streak for now)
    leaderboard = db.execute('''
        SELECT u.username, 
               SUM(s.duration_hours) as total_hours,
               AVG(s.focus_level) as avg_focus,
               (SUM(s.duration_hours) * 10 + AVG(s.focus_level)) as points
        FROM users u
        JOIN group_members m ON u.id = m.user_id
        LEFT JOIN study_sessions s ON u.id = s.user_id
        WHERE m.group_id = ?
        GROUP BY u.id
        ORDER BY points DESC
    ''', (group_id,)).fetchall()
    db.close()
    return jsonify([dict(l) for l in leaderboard]), 200

@app.route('/start_timer', methods=['POST'])
@login_required
def start_timer():
    data = request.get_json()
    subject = data.get('subject', 'General Study')
    db = get_db()
    db.execute('INSERT OR REPLACE INTO active_sessions (user_id, subject) VALUES (?, ?)',
               (session['user_id'], subject))
    db.commit()
    db.close()
    return jsonify({'message': 'Timer started'}), 200

@app.route('/stop_timer', methods=['POST'])
@login_required
def stop_timer():
    db = get_db()
    # Get the active session to calculate duration
    active = db.execute('SELECT * FROM active_sessions WHERE user_id = ?', (session['user_id'],)).fetchone()
    
    if active:
        # Calculate duration in hours
        start_time = datetime.strptime(active['start_time'], '%Y-%m-%d %H:%M:%S')
        now = datetime.now()
        duration_seconds = (now - start_time).total_seconds()
        duration_hours = round(max(duration_seconds / 3600, 0.1), 2)
        
        # Persist to study_sessions
        db.execute('''
            INSERT INTO study_sessions 
            (user_id, subject, start_time, end_time, duration_hours, self_rating, focus_level, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session['user_id'], 
            active['subject'], 
            active['start_time'], 
            now.strftime('%H:%M:%S'), 
            duration_hours,
            3, # Default rating for auto-logged sessions
            7, # Default focus for auto-logged sessions
            now.strftime('%Y-%m-%d')
        ))
        
    db.execute('DELETE FROM active_sessions WHERE user_id = ?', (session['user_id'],))
    db.commit()
    db.close()
    return jsonify({'message': 'Timer stopped and session saved'}), 200

@app.route('/get_sessions', methods=['GET'])
@login_required
def get_sessions():
    db = get_db()
    rows = db.execute('''SELECT * FROM study_sessions WHERE user_id=?
                         ORDER BY date DESC, created_at DESC''',
                      (session['user_id'],)).fetchall()
    db.close()
    sessions = [dict(r) for r in rows]
    return jsonify({'sessions': sessions}), 200

@app.route('/get_dashboard', methods=['GET'])
@login_required
def get_dashboard():
    db = get_db()
    uid = session['user_id']

    # Total hours
    total = db.execute('SELECT COALESCE(SUM(duration_hours),0) as total FROM study_sessions WHERE user_id=?', (uid,)).fetchone()
    total_hours = total['total']

    # Subjects
    subjects = db.execute('SELECT DISTINCT subject FROM study_sessions WHERE user_id=?', (uid,)).fetchall()
    subjects_list = [s['subject'] for s in subjects]

    # Avg focus
    avg_focus = db.execute('SELECT COALESCE(AVG(focus_level),0) as avg FROM study_sessions WHERE user_id=?', (uid,)).fetchone()
    avg_focus_val = round(avg_focus['avg'], 1)

    # Avg self rating
    avg_rating = db.execute('SELECT COALESCE(AVG(self_rating),0) as avg FROM study_sessions WHERE user_id=?', (uid,)).fetchone()
    productivity_score = round((avg_focus_val / 10) * 100, 1)

    # Weekly hours for last 7 days
    seven_days = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
    weekly_hours = []
    for day in seven_days:
        row = db.execute('SELECT COALESCE(SUM(duration_hours),0) as h FROM study_sessions WHERE user_id=? AND date=?',
                         (uid, day)).fetchone()
        weekly_hours.append(round(row['h'], 2))

    # Subject distribution
    subject_data = db.execute('''SELECT subject, COALESCE(SUM(duration_hours),0) as total_h
                                  FROM study_sessions WHERE user_id=?
                                  GROUP BY subject ORDER BY total_h DESC''', (uid,)).fetchall()
    subject_dist = [{'subject': r['subject'], 'hours': round(r['total_h'], 2)} for r in subject_data]

    # Role Suggestion Logic
    ROLE_MAPPING = {
        'python': 'Software Engineer',
        'java': 'Backend Developer',
        'javascript': 'Frontend Developer',
        'sql': 'Data Analyst',
        'database': 'Database Administrator',
        'machine learning': 'AI Specialist',
        'deep learning': 'AI Specialist',
        'data science': 'Data Scientist',
        'html': 'Web Developer',
        'css': 'Web Developer',
        'react': 'Frontend Developer',
        'node': 'Backend Developer',
        'aws': 'Cloud Engineer',
        'cloud': 'Cloud Engineer',
        'security': 'Cybersecurity Analyst',
        'networking': 'Network Engineer',
        'dsa': 'Software Engineer',
        'algorithms': 'Software Developer'
    }

    suggested_role = None
    if subject_dist:
        top_subject = subject_dist[0]['subject'].lower()
        for keyword, role in ROLE_MAPPING.items():
            if keyword in top_subject:
                suggested_role = role
                break
        
        # Generic fallback based on top subject if no mapping found
        if not suggested_role:
             suggested_role = f"{subject_dist[0]['subject']} Specialist"

    db.close()

    # Simple exam readiness estimate
    consistency = min(len([h for h in weekly_hours if h > 0]) / 7 * 100, 100)
    exam_readiness = round(
        (min(total_hours / 50, 1) * 30) +
        (productivity_score * 0.4) +
        (consistency * 0.3), 1
    )
    exam_readiness = min(exam_readiness, 100)

    return jsonify({
        'total_hours': round(total_hours, 2),
        'subjects_studied': len(subjects_list),
        'subjects': subjects_list,
        'productivity_score': productivity_score,
        'exam_readiness': exam_readiness,
        'avg_focus': avg_focus_val,
        'weekly_hours': weekly_hours,
        'weekly_labels': [d[5:] for d in seven_days],
        'subject_distribution': subject_dist,
        'suggested_role': suggested_role
    }), 200

# ── AI Analysis Routes ─────────────────────────────────────────────────────────
@app.route('/productivity_analysis', methods=['POST'])
@login_required
def productivity_analysis():
    data = request.get_json()
    uid = session['user_id']

    # Get user's session data for analysis
    db = get_db()
    sessions_data = db.execute('''SELECT * FROM study_sessions WHERE user_id=?
                                   ORDER BY date DESC''', (uid,)).fetchall()
    db.close()

    if not sessions_data:
        return jsonify({
            'status': 'no_data',
            'message': 'No study sessions found. Add sessions to analyze productivity.',
            'productive_hours': 0, 'unproductive_hours': 0,
            'productivity_percentage': 0, 'sessions': []
        }), 200

    total_hours = sum(s['duration_hours'] for s in sessions_data)
    high_focus = [s for s in sessions_data if s['focus_level'] >= 7]
    low_focus = [s for s in sessions_data if s['focus_level'] < 7]
    productive_hours = sum(s['duration_hours'] for s in high_focus)
    unproductive_hours = sum(s['duration_hours'] for s in low_focus)

    # Use ML model if available
    ml_result = None
    if productivity_model_data:
        model = productivity_model_data['model']
        features = productivity_model_data['features']
        feature_map = {
            'study_hours_per_day': data.get('study_hours', total_hours / max(len(sessions_data), 1)),
            'sleep_hours': data.get('sleep_hours', 7),
            'attendance_percentage': data.get('attendance', 80),
            'mental_health_rating': data.get('mental_health', 7),
            'motivation_level': data.get('motivation', 7),
            'time_management_score': data.get('time_management', 6),
            'stress_level': data.get('stress_level', 5),
            'exam_anxiety_score': data.get('exam_anxiety', 5),
            'exercise_frequency': data.get('exercise_frequency', 3),
            'social_media_hours': data.get('social_media_hours', 2),
            'screen_time': data.get('screen_time', 4)
        }
        input_vals = [[feature_map.get(f, 5) for f in features]]
        pred = model.predict(input_vals)[0]
        proba = model.predict_proba(input_vals)[0]
        ml_result = {
            'prediction': 'Productive' if pred == 1 else 'Needs Improvement',
            'confidence': round(max(proba) * 100, 1)
        }

    # Weak subjects (low focus or hours)
    subject_stats = {}
    for s in sessions_data:
        sub = s['subject']
        if sub not in subject_stats:
            subject_stats[sub] = {'hours': 0, 'focus_sum': 0, 'count': 0}
        subject_stats[sub]['hours'] += s['duration_hours']
        subject_stats[sub]['focus_sum'] += s['focus_level']
        subject_stats[sub]['count'] += 1

    weak_subjects = []
    recommendations = []
    for sub, stats in subject_stats.items():
        avg_focus = stats['focus_sum'] / stats['count']
        if avg_focus < 6 or stats['hours'] < 2:
            weak_subjects.append(sub)
            recommendations.append(f"📚 Spend more time studying {sub} — avg focus was {avg_focus:.1f}/10")

    return jsonify({
        'status': 'success',
        'productive_hours': round(productive_hours, 2),
        'unproductive_hours': round(unproductive_hours, 2),
        'productivity_percentage': round((productive_hours / max(total_hours, 0.1)) * 100, 1),
        'weak_subjects': weak_subjects,
        'recommendations': recommendations,
        'ml_result': ml_result,
        'sessions': [dict(s) for s in sessions_data[:10]]
    }), 200

@app.route('/placement_prediction', methods=['POST'])
@login_required
def placement_prediction():
    data = request.get_json()
    
    # Extract inputs directly from the request
    study_hours = float(data.get('study_hours', 4))
    social_media = float(data.get('social_media_hours', 2))
    stress_level = int(data.get('stress_level', 5))
    time_management = int(data.get('time_management', 5))
    communication_skills = int(data.get('communication_skills', 5))

    user_input = {
        'study_hours_per_day': study_hours,
        'social_media_hours': social_media,
        'stress_level': stress_level,
        'time_management_score': time_management
    }

    ready_status = None
    if placement_model_data and scaler:
        model = placement_model_data['model']
        features = placement_model_data['features']
        input_vals = [[user_input.get(f, 5) for f in features]]
        try:
            input_scaled = scaler.transform(input_vals)
            pred = model.predict(input_scaled)[0]
            
            # Incorporate communication skills as a weighted factor (heuristic)
            # If communication is very low, it might override a 'Ready' status
            if pred == 1 and communication_skills < 4:
                ready_status = 'Not Ready (Improve Soft Skills)'
            elif pred == 0 and communication_skills >= 8 and study_hours >= 5:
                ready_status = 'Potentially Ready (Strong Communication)'
            else:
                ready_status = 'Ready' if pred == 1 else 'Not Ready'
        except Exception as e:
            print(f"[WARN] Prediction error: {e}")

    if ready_status is None:
        # Fallback logic
        is_technically_ready = study_hours >= 4 and stress_level <= 6 and time_management >= 5
        is_soft_skill_ready = communication_skills >= 5
        ready_status = 'Ready' if is_technically_ready and is_soft_skill_ready else 'Not Ready'

    advice = "Great job! Your profile looks strong for upcoming placements. Focus on maintaining this balance." if 'Ready' in ready_status else \
             "Consider improving your technical focus or soft skills. Balanced preparation is key for placements."

    return jsonify({
        'status': ready_status,
        'advice': advice,
        'input_summary': user_input,
        'communication_skills': communication_skills
    }), 200

@app.route('/placement_roadmap', methods=['POST'])
@login_required
def placement_roadmap():
    data = request.get_json()
    role = data.get('role', 'Software Engineer')
    
    # Define role templates to handle the exhaustive 60+ list
    role_templates = {
        'Software Development': {
            'roadmap': [
                {'step': 'Coding Foundations', 'desc': 'Master DSA and OOPS in Java, C++, or Python.', 'priority': 'High'},
                {'step': 'Tech Stack Mastery', 'desc': 'Learn relevant frameworks (React, Node, Django, Flutter, etc.).', 'priority': 'High'},
                {'step': 'Project Development', 'desc': 'Build robust applications and document on GitHub.', 'priority': 'Medium'},
                {'step': 'Quality & Scale', 'desc': 'Learn testing, clean code, and basic system design.', 'priority': 'Medium'}
            ],
            'books': [
                {'title': 'Cracking the Coding Interview', 'author': 'Gayle Laakmann McDowell'},
                {'title': 'Clean Code', 'author': 'Robert C. Martin'}
            ],
            'playlists': [
                {'name': 'Striver\'s SDE Sheet', 'link': 'https://www.youtube.com/c/takeUforward'},
                {'name': 'FreeCodeCamp Development', 'link': 'https://www.youtube.com/c/Freecodecamp'}
            ]
        },
        'Data & AI': {
            'roadmap': [
                {'step': 'Math & Stats', 'desc': 'Linear Algebra, Calculus, and Probability.', 'priority': 'High'},
                {'step': 'Data Engineering', 'desc': 'Master SQL, Python (Pandas/NumPy), and ETL basics.', 'priority': 'High'},
                {'step': 'ML/AI Theory', 'desc': 'Understand algorithms, training cycles, and model evaluation.', 'priority': 'High'},
                {'step': 'Deployment', 'desc': 'Learn Flask/FastAPI for model serving and cloud basics.', 'priority': 'Medium'}
            ],
            'books': [
                {'title': 'Hands-On ML with Scikit-Learn', 'author': 'Aurélien Géron'},
                {'title': 'Deep Learning (Adaptive Computation)', 'author': 'Ian Goodfellow'}
            ],
            'playlists': [
                {'name': 'Krish Naik AI', 'link': 'https://www.youtube.com/user/krishnaik06'},
                {'name': 'DeepLearning.AI', 'link': 'https://www.youtube.com/c/Deeplearningai'}
            ]
        },
        'Infrastructure & Security': {
            'roadmap': [
                {'step': 'Networking Basics', 'desc': 'Understand OSI levels, TCP/IP, and DNS.', 'priority': 'High'},
                {'step': 'Cloud Foundations', 'desc': 'AWS/Azure/GCP certifications and IAM.', 'priority': 'High'},
                {'step': 'Security/SRE', 'desc': 'Linux hardening, monitoring, and incident response.', 'priority': 'High'},
                {'step': 'Automation', 'desc': 'Master Bash scripting, Terraform, or Ansible.', 'priority': 'Medium'}
            ],
            'books': [
                {'title': 'Cloud Native DevOps with Kubernetes', 'author': 'John Arundel'},
                {'title': 'Hacking: The Art of Exploitation', 'author': 'Jon Erickson'}
            ],
            'playlists': [
                {'name': 'TechWorld with Nana', 'link': 'https://www.youtube.com/c/TechWorldwithNana'},
                {'name': 'NetworkChuck', 'link': 'https://www.youtube.com/c/NetworkChuck'}
            ]
        },
        'Product & Management': {
            'roadmap': [
                {'step': 'Market Insights', 'desc': 'Case studies, guesstimates, and competitive analysis.', 'priority': 'High'},
                {'step': 'Business Logic', 'desc': 'Understand KPIs, ROI, and product life cycle.', 'priority': 'High'},
                {'step': 'Strategic Planning', 'desc': 'Roadmap creation and stakeholder coordination.', 'priority': 'High'},
                {'step': 'User Centricity', 'desc': 'Learn UX principles and customer feedback loops.', 'priority': 'Medium'}
            ],
            'books': [
                {'title': 'Inspired: How to Create Tech Products', 'author': 'Marty Cagan'},
                {'title': 'The Lean Startup', 'author': 'Eric Ries'}
            ],
            'playlists': [
                {'name': 'Exponent Career', 'link': 'https://www.youtube.com/c/ExponentTV'},
                {'name': 'Product School', 'link': 'https://www.youtube.com/c/ProductSchoolSanFrancisco'}
            ]
        },
        'Marketing & Design': {
            'roadmap': [
                {'step': 'Visual Principles', 'desc': 'Color theory, typography, and layout mastery.', 'priority': 'High'},
                {'step': 'Tool Proficiency', 'desc': 'Master Figma, Adobe Suite, or SEO tools.', 'priority': 'High'},
                {'step': 'Brand Strategy', 'desc': 'Identify target audiences and messaging.', 'priority': 'Medium'},
                {'step': 'Execution', 'desc': 'Portfolio building and campaign analysis.', 'priority': 'Medium'}
            ],
            'books': [
                {'title': 'Don\'t Make Me Think', 'author': 'Steve Krug'},
                {'title': 'Building a StoryBrand', 'author': 'Donald Miller'}
            ],
            'playlists': [
                {'name': 'The Futur', 'link': 'https://www.youtube.com/c/TheFuturishere'},
                {'name': 'Ahrefs Marketing Academy', 'link': 'https://www.youtube.com/c/AhrefsCom'}
            ]
        },
        'HR & Operations': {
            'roadmap': [
                {'step': 'People Skills', 'desc': 'Effective communication and conflict resolution.', 'priority': 'High'},
                {'step': 'Recruitment Tech', 'desc': 'ATS systems and technical vetting basics.', 'priority': 'High'},
                {'step': 'Process Improvement', 'desc': 'Lean management and operational efficiency.', 'priority': 'Medium'},
                {'step': 'Legal/Compliance', 'desc': 'Labor laws and corporate policies.', 'priority': 'Medium'}
            ],
            'books': [
                {'title': 'Work Rules!', 'author': 'Laszlo Bock'},
                {'title': 'The HR Scorecard', 'author': 'Brian Becker'}
            ],
            'playlists': [
                {'name': 'Society for HR Management', 'link': 'https://www.youtube.com/c/SHRM'},
                {'name': 'Project Management Institute', 'link': 'https://www.youtube.com/c/pmi'}
            ]
        }
    }

    # Role Grouping Map
    role_map = {
        # Technical - Dev
        'Software Engineer': 'Software Development', 'Software Developer': 'Software Development',
        'Frontend Developer': 'Software Development', 'Backend Developer': 'Software Development',
        'Full Stack Developer': 'Software Development', 'Web Developer': 'Software Development',
        'Mobile App Developer': 'Software Development', 'Game Developer': 'Software Development',
        'QA Engineer': 'Software Development', 'Automation Test Engineer': 'Software Development',
        'Manual Tester': 'Software Development', 'Performance Tester': 'Software Development',
        'Embedded Systems Engineer': 'Software Development', 'IoT Engineer': 'Software Development',
        'Blockchain Developer': 'Software Development', 'AR/VR Developer': 'Software Development',
        'Solutions Architect': 'Software Development',
        
        # Technical - Data/AI
        'Data Scientist': 'Data & AI', 'Data Analyst': 'Data & AI', 
        'Machine Learning Engineer': 'Data & AI', 'AI Engineer': 'Data & AI',
        'Deep Learning Engineer': 'Data & AI', 'Data Engineer': 'Data & AI',
        'Database Administrator (DBA)': 'Data & AI', 'Database Developer': 'Data & AI',
        
        # Technical - Infra/Security
        'Cloud Engineer': 'Infrastructure & Security', 'DevOps Engineer': 'Infrastructure & Security',
        'Site Reliability Engineer (SRE)': 'Infrastructure & Security', 'System Administrator': 'Infrastructure & Security',
        'Network Engineer': 'Infrastructure & Security', 'Cybersecurity Analyst': 'Infrastructure & Security',
        'Security Engineer': 'Infrastructure & Security', 'Ethical Hacker': 'Infrastructure & Security',
        'Penetration Tester': 'Infrastructure & Security', 'Information Security Analyst': 'Infrastructure & Security',
        
        # Non-Technical - Management
        'Project Manager': 'Product & Management', 'Program Manager': 'Product & Management',
        'Product Manager': 'Product & Management', 'Product Owner': 'Product & Management',
        'Operations Manager': 'Product & Management', 'Business Analyst': 'Product & Management',
        'Data Product Manager': 'Product & Management',
        
        # Non-Technical - Consulting/Support
        'IT Consultant': 'Product & Management', 'Technical Support Engineer': 'Product & Management',
        'Customer Success Manager': 'Product & Management', 'Pre-Sales Engineer': 'Product & Management',
        'Solutions Consultant': 'Product & Management', 'Technology Consultant': 'Product & Management',
        
        # Non-Technical - Marketing/Design
        'Digital Marketing Specialist': 'Marketing & Design', 'SEO Specialist': 'Marketing & Design',
        'Product Marketing Manager': 'Marketing & Design', 'Sales Executive (IT Products)': 'Marketing & Design',
        'UI Designer': 'Marketing & Design', 'UX Designer': 'Marketing & Design',
        'Product Designer': 'Marketing & Design', 'Graphic Designer': 'Marketing & Design',
        
        # Non-Technical - HR/Content
        'HR Manager': 'HR & Operations', 'Technical Recruiter': 'HR & Operations',
        'Training Manager': 'HR & Operations', 'Technical Writer': 'HR & Operations',
        'Content Writer': 'HR & Operations'
    }

    template_key = role_map.get(role, 'Software Development')
    role_data = role_templates.get(template_key)
    
    return jsonify({
        'role': role,
        'roadmap': role_data['roadmap'],
        'books': role_data['books'],
        'playlists': role_data['playlists'],
        'guide': [
            "● Focus on role-specific core competencies daily.",
            "● Build a strong portfolio aligned with chosen category.",
            "● Network with professionals currently in the target role.",
            "● Practice domain-specific mock interviews and case studies."
        ]
    }), 200

@app.route('/generate_study_plan', methods=['POST'])
@login_required
def generate_study_plan():
    data = request.get_json()
    uid = session['user_id']

    db = get_db()
    sessions_data = db.execute('''SELECT subject, SUM(duration_hours) as total_h,
                                          AVG(focus_level) as avg_focus
                                   FROM study_sessions WHERE user_id=?
                                   GROUP BY subject''', (uid,)).fetchall()
    db.close()

    subjects_input = data.get('subjects', [])
    daily_hours = float(data.get('daily_hours', 4))
    days_until_exam = int(data.get('days_until_exam', 14))

    # Combine from DB and user input
    subject_data = {}
    for row in sessions_data:
        subject_data[row['subject']] = {
            'hours': row['total_h'],
            'focus': row['avg_focus'],
            'priority': 'high' if row['avg_focus'] < 6 or row['total_h'] < 3 else 'normal'
        }

    for sub in subjects_input:
        if sub not in subject_data:
            subject_data[sub] = {'hours': 0, 'focus': 5, 'priority': 'high'}

    if not subject_data:
        return jsonify({'error': 'No subjects found. Add study sessions or provide subjects.'}), 400

    # Sort by priority (weak subjects get more time)
    sorted_subs = sorted(subject_data.items(),
                         key=lambda x: (x[1]['priority'] == 'high', -x[1]['focus']),
                         reverse=True)

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plan = []

    sub_cycle = sorted_subs * (days_until_exam // 7 + 2)
    for i, day in enumerate(days[:min(days_until_exam, 7)]):
        day_plan = {'day': day, 'sessions': []}
        remaining = daily_hours
        si = i * 2
        while remaining > 0 and si < len(sub_cycle):
            sub_name, sub_info = sub_cycle[si % len(sub_cycle)]
            time_slot = min(2 if sub_info['priority'] == 'high' else 1.5, remaining)
            day_plan['sessions'].append({
                'subject': sub_name,
                'hours': time_slot,
                'type': '🔥 Priority' if sub_info['priority'] == 'high' else '📖 Regular',
                'note': f"Focus score: {sub_info['focus']:.1f}/10"
            })
            remaining -= time_slot
            si += 1

        if remaining > 0.5:
            day_plan['sessions'].append({'subject': 'Revision & Practice', 'hours': remaining, 'type': '📝 Review', 'note': ''})

        plan.append(day_plan)

    return jsonify({
        'plan': plan,
        'total_subjects': len(subject_data),
        'daily_hours': daily_hours,
        'days': min(days_until_exam, 7)
    }), 200

# ── Notifications ──────────────────────────────────────────────────────────────
@app.route('/get_notifications', methods=['GET'])
@login_required
def get_notifications():
    db = get_db()
    notifs = db.execute('''SELECT * FROM notifications WHERE user_id=?
                           ORDER BY created_at DESC LIMIT 20''',
                        (session['user_id'],)).fetchall()
    unread = db.execute('SELECT COUNT(*) as cnt FROM notifications WHERE user_id=? AND is_read=0',
                        (session['user_id'],)).fetchone()
    db.close()
    return jsonify({
        'notifications': [dict(n) for n in notifs],
        'unread_count': unread['cnt']
    }), 200

@app.route('/mark_notifications_read', methods=['POST'])
@login_required
def mark_notifications_read():
    db = get_db()
    db.execute('UPDATE notifications SET is_read=1 WHERE user_id=?', (session['user_id'],))
    db.commit()
    db.close()
    return jsonify({'message': 'All notifications marked as read'}), 200

# ── Profile ────────────────────────────────────────────────────────────────────
@app.route('/get_profile', methods=['GET'])
@login_required
def get_profile():
    db = get_db()
    uid = session['user_id']
    user = db.execute('SELECT id, username, email, created_at FROM users WHERE id=?', (uid,)).fetchone()
    total = db.execute('SELECT COALESCE(SUM(duration_hours),0) as t FROM study_sessions WHERE user_id=?', (uid,)).fetchone()
    sess_count = db.execute('SELECT COUNT(*) as c FROM study_sessions WHERE user_id=?', (uid,)).fetchone()
    subjects = db.execute('SELECT COUNT(DISTINCT subject) as s FROM study_sessions WHERE user_id=?', (uid,)).fetchone()
    db.close()
    return jsonify({
        'user': dict(user),
        'total_hours': round(total['t'], 2),
        'session_count': sess_count['c'],
        'subjects_count': subjects['s']
    }), 200

# ── Weekly Progress ────────────────────────────────────────────────────────────
@app.route('/weekly_progress', methods=['GET'])
@login_required
def weekly_progress():
    db = get_db()
    uid = session['user_id']

    # Last 4 weeks
    weeks_data = []
    for w in range(4):
        start = (datetime.now() - timedelta(weeks=w+1)).strftime('%Y-%m-%d')
        end = (datetime.now() - timedelta(weeks=w)).strftime('%Y-%m-%d')
        row = db.execute('''SELECT COALESCE(SUM(duration_hours),0) as h,
                                   COALESCE(AVG(focus_level),0) as f
                            FROM study_sessions WHERE user_id=? AND date>=? AND date<?''',
                         (uid, start, end)).fetchone()
        weeks_data.append({
            'week': f'Week -{w+1}' if w > 0 else 'This Week',
            'hours': round(row['h'], 2),
            'avg_focus': round(row['f'], 1)
        })

    weeks_data.reverse()

    # Subject breakdown for this week
    week_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    subject_rows = db.execute('''SELECT subject, SUM(duration_hours) as h, AVG(focus_level) as f
                                  FROM study_sessions WHERE user_id=? AND date>=?
                                  GROUP BY subject ORDER BY h DESC''',
                               (uid, week_start)).fetchall()
    db.close()

    return jsonify({
        'weeks': weeks_data,
        'subjects_this_week': [{'subject': r['subject'], 'hours': round(r['h'],2), 'avg_focus': round(r['f'],1)} for r in subject_rows]
    }), 200

# ── Helper: auto notifications ─────────────────────────────────────────────────
def _check_and_create_notifications(user_id):
    db = get_db()
    # Check weekly hours
    week_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    week_hours = db.execute('''SELECT COALESCE(SUM(duration_hours),0) as h
                               FROM study_sessions WHERE user_id=? AND date>=?''',
                            (user_id, week_start)).fetchone()
    if week_hours['h'] < 5:
        create_notification(user_id, "⚠️ You've studied less than 5 hours this week. Try to increase your study time!", 'warning')

    # Check weak subjects
    weak = db.execute('''SELECT subject FROM study_sessions WHERE user_id=?
                         GROUP BY subject HAVING AVG(focus_level) < 5 OR SUM(duration_hours) < 2''',
                      (user_id,)).fetchall()
    for s in weak:
        create_notification(user_id, f"📚 {s['subject']} needs more attention. Your focus/hours are low.", 'warning')
    db.close()

# ── Subject Analysis ───────────────────────────────────────────────────────────
@app.route('/subject_analysis', methods=['GET'])
@login_required
def subject_analysis():
    db = get_db()
    uid = session['user_id']
    rows = db.execute('''SELECT subject,
                                SUM(duration_hours) as total_hours,
                                AVG(focus_level) as avg_focus,
                                AVG(self_rating) as avg_rating,
                                COUNT(*) as session_count
                         FROM study_sessions WHERE user_id=?
                         GROUP BY subject ORDER BY total_hours DESC''', (uid,)).fetchall()
    db.close()

    subjects = []
    for r in rows:
        avg_f = r['avg_focus'] or 0
        status = 'Strong' if avg_f >= 7 and r['total_hours'] >= 5 else \
                 'Average' if avg_f >= 5 or r['total_hours'] >= 2 else 'Weak'
        subjects.append({
            'subject': r['subject'],
            'total_hours': round(r['total_hours'], 2),
            'avg_focus': round(avg_f, 1),
            'avg_rating': round(r['avg_rating'] or 0, 1),
            'session_count': r['session_count'],
            'status': status,
            'recommendation': f"Keep up the good work with {r['subject']}!" if status == 'Strong'
                              else f"Spend more time on {r['subject']} to improve." if status == 'Weak'
                              else f"Consistent practice will strengthen {r['subject']}."
        })

    return jsonify({'subjects': subjects}), 200


# ── Tutorial Assessment Routes ───────────────────────────────────────────────
@app.route('/submit_assessment', methods=['POST'])
@login_required
def submit_assessment():
    data = request.get_json()
    uid = session['user_id']
    subject = data.get('subject', '').strip()
    score = int(data.get('score', 0))
    total = int(data.get('total', 0))
    time_per_question = data.get('time_per_question', []) # List of seconds per question

    if not subject or total == 0:
        return jsonify({'error': 'Subject and total questions are required'}), 400

    percentage = (score / total) * 100
    avg_time = sum(time_per_question) / len(time_per_question) if time_per_question else 0

    # Prediction Logic: Ready if score >= 80% AND avg_time < 30 seconds
    if percentage >= 80 and (avg_time > 0 and avg_time <= 30):
        readiness_status = 'Ready for Placement'
        advice = f"Excellent! You mastered {subject.upper()} with {percentage}% accuracy and quick response times. Your technical readiness is high."
    elif percentage >= 80:
        readiness_status = 'Technically Ready (Improve Speed)'
        advice = f"Good accuracy, but your response time ({avg_time:.1s}s avg) is slightly slow. For placements, speed is as important as accuracy."
    else:
        readiness_status = 'Need to Improve'
        advice = f"You scored {percentage}%. We recommend reviewing the {subject} tutorial again before attempting the assessment for placement readiness."

    # Persistence
    db = get_db()
    db.execute('''INSERT INTO assessments 
                  (user_id, subject, score, total, percentage, avg_time_per_question, readiness_status)
                  VALUES (?, ?, ?, ?, ?, ?, ?)''',
               (uid, subject, score, total, percentage, avg_time, readiness_status))
    db.commit()
    db.close()

    create_notification(uid, f"Assessment Complete: {subject.capitalize()} - {readiness_status}", 'success' if 'Ready' in readiness_status else 'info')

    return jsonify({
        'status': readiness_status,
        'percentage': percentage,
        'avg_time': round(avg_time, 2),
        'advice': advice
    }), 200

# ── AI Hybrid Chatbot Route ──────────────────────────────────────────────────
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    query_lower = query.lower()
    found_answer = None
    
    # 1. Check Project FAQ
    for item in faq_data:
        for kw in item.get('keywords', []):
            if kw in query_lower:
                found_answer = item['answer']
                source = "FAQ"
                break
        if found_answer:
            break
            
    # 2. Fallback to Gemini AI
    if not found_answer:
        found_answer, source = real_llm_query(query)
        
    return jsonify({
        'answer': found_answer,
        'source': source,
        'query': query
    }), 200

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    load_models()
    load_faq()

    port = int(os.environ.get("PORT", 5000))

    print(f"\n🚀 AI Study Tracker Backend starting on port {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
