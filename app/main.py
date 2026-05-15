"""
电影推荐系统 Web 应用 - 基于 Flask
"""
import os
import sys
import sqlite3
import hashlib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, g

app = Flask(__name__)
app.secret_key = 'recommend-movie-secret-key-2024'
app.config['DATABASE'] = os.path.join(os.path.dirname(__file__), 'movie_app.db')

# 模型路径
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved')

# 全局变量加载模型
user_embeddings = None
item_embeddings = None
item_id_map = {}      # recbole_id -> movie_id
movie_info = {}       # movie_id -> {title, genres, ...}
user_id_map = {}      # recbole_id -> user_id


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    db_path = app.config['DATABASE']
    db = sqlite3.connect(db_path)
    # 确保 recbole_user_id 列存在（兼容已有数据库）
    try:
        db.execute('ALTER TABLE users ADD COLUMN recbole_user_id INTEGER DEFAULT NULL')
    except:
        pass  # 列已存在
    db.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        recbole_user_id INTEGER DEFAULT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    db.execute('''CREATE TABLE IF NOT EXISTS movies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        movie_id INTEGER UNIQUE NOT NULL,
        title TEXT NOT NULL,
        genres TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    db.execute('''CREATE TABLE IF NOT EXISTS ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        movie_id INTEGER NOT NULL,
        rating REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (movie_id) REFERENCES movies(id)
    )''')
    db.execute('''CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        movie_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (movie_id) REFERENCES movies(id)
    )''')
    # 管理员
    admin_pwd = hashlib.sha256('admin123'.encode()).hexdigest()
    db.execute(
        'INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)',
        ('admin', admin_pwd, 'admin')
    )
    # 数据集用户 user_1 ~ user_943，密码统一 123456
    default_pwd = hashlib.sha256('123456'.encode()).hexdigest()
    for i in range(1, 944):
        db.execute(
            'INSERT OR IGNORE INTO users (username, password, role, recbole_user_id) VALUES (?, ?, ?, ?)',
            (f'user_{i}', default_pwd, 'user', i - 1)
        )
    db.commit()
    db.close()


def load_model():
    """加载训练好的模型嵌入和元数据"""
    global user_embeddings, item_embeddings, item_id_map, movie_info, user_id_map
    try:
        user_path = os.path.join(MODEL_DIR, 'user_emb.npy')
        item_path = os.path.join(MODEL_DIR, 'item_emb.npy')
        if not os.path.exists(user_path) or not os.path.exists(item_path):
            # 使用随机嵌入作为占位
            user_embeddings = np.random.randn(944, 64).astype(np.float32)
            item_embeddings = np.random.randn(1683, 64).astype(np.float32)
        else:
            user_embeddings = np.load(user_path)
            item_embeddings = np.load(item_path)
        print(f'Model loaded: users={user_embeddings.shape}, items={item_embeddings.shape}')
    except Exception as e:
        print(f'Model load failed, using random embeddings: {e}')
        user_embeddings = np.random.randn(944, 64).astype(np.float32)
        item_embeddings = np.random.randn(1683, 64).astype(np.float32)


def load_movie_data():
    """加载 MovieLens-100k 元数据（兼容 RecBole .item/.user 格式）"""
    global item_id_map, movie_info, user_id_map
    try:
        import recbole
        ds_dir = os.path.join(os.path.dirname(recbole.__file__), 'dataset_example', 'ml-100k')
        if not os.path.exists(ds_dir):
            import glob as _glob
            for c in _glob.glob(os.path.expanduser('~') + '/**/ml-100k', recursive=True):
                if os.path.isdir(c):
                    ds_dir = c
                    break

        # 读取 RecBole 格式的 .item 文件 (TSV, header: field:type)
        item_file = os.path.join(ds_dir, 'ml-100k.item')
        if os.path.exists(item_file):
            with open(item_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # 跳过 header 行
            for idx, line in enumerate(lines[1:]):
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    try:
                        mid = int(parts[0])
                        title = parts[1]
                        genres = parts[3] if len(parts) > 3 else 'Unknown'
                        movie_info[mid] = {'title': title, 'genres': genres}
                        item_id_map[idx] = mid
                    except:
                        pass

        # 读取 RecBole 格式的 .user 文件
        user_file = os.path.join(ds_dir, 'ml-100k.user')
        if os.path.exists(user_file):
            with open(user_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                parts = line.strip().split('\t')
                if parts:
                    try:
                        uid = int(parts[0])
                        user_id_map[idx] = uid
                    except:
                        pass

        print(f'Loaded {len(movie_info)} movies, {len(user_id_map)} users')
    except Exception as e:
        print(f'Failed to load movie data: {e}')
        for i in range(1682):
            item_id_map[i] = i + 1
            movie_info[i + 1] = {'title': f'Movie {i + 1}', 'genres': 'Unknown'}


def populate_movies_table():
    """将电影元数据同步到数据库"""
    db = get_db()
    for mid, info in movie_info.items():
        db.execute(
            'INSERT OR IGNORE INTO movies (movie_id, title, genres) VALUES (?, ?, ?)',
            (mid, info['title'], info.get('genres', 'Unknown'))
        )
    db.commit()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ==================== 认证 ====================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录')
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('需要管理员权限')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated


# ==================== 推荐引擎 ====================

def _reverse_item_map():
    return {v: k for k, v in item_id_map.items()}


def _reverse_user_map():
    return {v: k for k, v in user_id_map.items()}


def get_user_embedding(user_id):
    """获取用户个性化嵌入向量
    优先级:
      1. user 有 recbole_user_id → 直接用训练好的嵌入
      2. user 有评分记录 → 加权合成嵌入
      3. 新用户 → 全局平均嵌入（冷启动）
    返回: (embedding, source, rating_count)
    """
    global user_embeddings, item_embeddings
    if user_embeddings is None:
        load_model()

    # 1. 检查是否是数据集内用户
    try:
        db = get_db()
        row = db.execute(
            'SELECT recbole_user_id FROM users WHERE id = ?', (user_id,)
        ).fetchone()
        if row and row['recbole_user_id'] is not None:
            rid = row['recbole_user_id']
            if rid < len(user_embeddings):
                return user_embeddings[rid], 'pretrained', 0
    except:
        pass

    # 2. 根据评分记录加权合成
    try:
        db = get_db()
        ratings = db.execute(
            'SELECT movie_id, rating FROM ratings WHERE user_id = ?', (user_id,)
        ).fetchall()
        if ratings:
            rmap = _reverse_item_map()
            weighted_emb = np.zeros(user_embeddings.shape[1], dtype=np.float32)
            total_weight = 0.0
            for r in ratings:
                rid = rmap.get(r['movie_id'])
                if rid is not None and rid < len(item_embeddings):
                    w = float(r['rating']) - 2.5
                    weighted_emb += w * item_embeddings[rid]
                    total_weight += abs(w)
            if total_weight > 1e-8:
                return weighted_emb / total_weight, 'rated', len(ratings)
    except:
        pass

    # 3. 冷启动
    return user_embeddings.mean(axis=0), 'cold', 0


def get_recommendations(user_id=None, top_k=20):
    """为用户生成个性化推荐列表
    返回: (recommendations, embedding_source)
      - embedding_source: 'rated' (评分个性化) / 'cold' (冷启动)
    """
    global user_embeddings, item_embeddings

    if user_embeddings is None or item_embeddings is None:
        load_model()

    u_emb, source, rating_count = get_user_embedding(user_id)

    # 计算评分
    scores = np.dot(item_embeddings, u_emb)

    # 排除已评分电影
    exclude_ids = set()
    if user_id:
        try:
            db = get_db()
            rated = db.execute(
                'SELECT movie_id FROM ratings WHERE user_id = ?', (user_id,)
            ).fetchall()
            rmap = _reverse_item_map()
            for r in rated:
                rid = rmap.get(r['movie_id'])
                if rid is not None:
                    exclude_ids.add(rid)
        except:
            pass

    # 获取 top-K
    sorted_indices = np.argsort(scores)[::-1]
    recommendations = []
    for idx in sorted_indices:
        if len(recommendations) >= top_k:
            break
        if idx in exclude_ids:
            continue
        mid = item_id_map.get(idx, idx + 1)
        info = movie_info.get(mid, {'title': f'Movie {mid}', 'genres': 'Unknown'})
        recommendations.append({
            'id': mid,
            'title': info['title'],
            'genres': info.get('genres', 'Unknown'),
            'score': float(scores[idx])
        })
    return recommendations, source


def get_similar_movies(movie_id, top_k=10):
    """获取相似电影"""
    global item_embeddings
    if item_embeddings is None:
        load_model()

    # 找到电影的 recbole 索引
    reverse_item_map = {v: k for k, v in item_id_map.items()}
    idx = reverse_item_map.get(movie_id)
    if idx is None or idx >= len(item_embeddings):
        return []

    m_emb = item_embeddings[idx]
    scores = np.dot(item_embeddings, m_emb)
    sorted_indices = np.argsort(scores)[::-1]

    similar = []
    for i in sorted_indices:
        if len(similar) >= top_k + 1:
            break
        if i == idx:
            continue
        rid = item_id_map.get(i, i + 1)
        if rid == movie_id:
            continue
        info = movie_info.get(rid, {'title': f'Movie {rid}', 'genres': 'Unknown'})
        similar.append({
            'id': rid,
            'title': info['title'],
            'genres': info.get('genres', 'Unknown'),
            'score': float(scores[i])
        })
    return similar


# ==================== 路由 ====================

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if session.get('role') == 'admin':
        return redirect(url_for('admin'))

    # 获取推荐
    user_id = session['user_id']
    recommendations, source = get_recommendations(user_id=user_id, top_k=20)

    # 获取高评分电影
    db = get_db()
    popular = db.execute(
        'SELECT m.movie_id, m.title, m.genres, AVG(r.rating) as avg_rating, COUNT(r.id) as cnt '
        'FROM ratings r JOIN movies m ON r.movie_id = m.movie_id '
        'GROUP BY m.movie_id ORDER BY avg_rating DESC LIMIT 10'
    ).fetchall()

    return render_template('index.html',
                           recommendations=recommendations,
                           popular=popular,
                           source=source,
                           username=session['username'])


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('请输入用户名和密码')
            return render_template('login.html')

        db = get_db()
        user = db.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()

        if user and user['password'] == hash_password(password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            flash(f'欢迎回来, {username}!')
            return redirect(url_for('index'))
        else:
            flash('用户名或密码错误')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        if not username or not password:
            flash('请填写所有字段')
            return render_template('register.html')

        if password != confirm:
            flash('两次密码不一致')
            return render_template('register.html')

        if len(username) < 3:
            flash('用户名至少3个字符')
            return render_template('register.html')

        db = get_db()
        try:
            db.execute(
                'INSERT INTO users (username, password) VALUES (?, ?)',
                (username, hash_password(password))
            )
            db.commit()
        except sqlite3.IntegrityError:
            flash('用户名已存在')
            return render_template('register.html')

        flash('注册成功，请登录')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/search')
@login_required
def search():
    q = request.args.get('q', '').strip()
    genre = request.args.get('genre', '').strip()
    results = []

    if q or genre:
        db = get_db()
        query = 'SELECT * FROM movies WHERE 1=1'
        params = []

        if q:
            query += ' AND title LIKE ?'
            params.append(f'%{q}%')
        if genre:
            query += ' AND genres LIKE ?'
            params.append(f'%{genre}%')

        query += ' LIMIT 50'
        results = db.execute(query, params).fetchall()

    # 获取所有类型
    db = get_db()
    all_genres = set()
    for row in db.execute('SELECT genres FROM movies').fetchall():
        for g in row['genres'].split('|'):
            if g.strip():
                all_genres.add(g.strip())

    return render_template('search.html',
                           movies=results,
                           query=q,
                           selected_genre=genre,
                           all_genres=sorted(all_genres))


@app.route('/movie/<int:movie_id>')
@login_required
def movie_detail(movie_id):
    db = get_db()
    movie = db.execute('SELECT * FROM movies WHERE movie_id = ?', (movie_id,)).fetchone()
    if not movie:
        flash('电影不存在')
        return redirect(url_for('index'))

    # 用户评分
    user_rating = None
    if 'user_id' in session:
        r = db.execute(
            'SELECT rating FROM ratings WHERE user_id = ? AND movie_id = ?',
            (session['user_id'], movie_id)
        ).fetchone()
        if r:
            user_rating = r['rating']

    # 平均评分
    avg_row = db.execute(
        'SELECT AVG(rating) as avg_rating, COUNT(*) as cnt FROM ratings WHERE movie_id = ?',
        (movie_id,)
    ).fetchone()
    avg_rating = avg_row['avg_rating'] if avg_row['cnt'] > 0 else None
    rating_count = avg_row['cnt']

    # 评论
    reviews = db.execute(
        'SELECT r.*, u.username FROM reviews r JOIN users u ON r.user_id = u.id '
        'WHERE r.movie_id = ? ORDER BY r.created_at DESC',
        (movie_id,)
    ).fetchall()

    # 相似电影
    similar = get_similar_movies(movie_id, top_k=10)

    return render_template('movie_detail.html',
                           movie=movie,
                           user_rating=user_rating,
                           avg_rating=avg_rating,
                           rating_count=rating_count,
                           reviews=reviews,
                           similar=similar)


@app.route('/rate/<int:movie_id>', methods=['POST'])
@login_required
def rate_movie(movie_id):
    rating = request.form.get('rating', type=float)
    if not rating or rating < 1 or rating > 5:
        flash('评分需要在1-5之间')
        return redirect(url_for('movie_detail', movie_id=movie_id))

    db = get_db()
    db.execute(
        'INSERT OR REPLACE INTO ratings (user_id, movie_id, rating, created_at) '
        'VALUES (?, ?, ?, CURRENT_TIMESTAMP)',
        (session['user_id'], movie_id, rating)
    )
    db.commit()
    flash('评分成功!')
    return redirect(url_for('movie_detail', movie_id=movie_id))


@app.route('/review/<int:movie_id>', methods=['POST'])
@login_required
def review_movie(movie_id):
    content = request.form.get('content', '').strip()
    if not content:
        flash('评论内容不能为空')
        return redirect(url_for('movie_detail', movie_id=movie_id))

    db = get_db()
    db.execute(
        'INSERT INTO reviews (user_id, movie_id, content) VALUES (?, ?, ?)',
        (session['user_id'], movie_id, content)
    )
    db.commit()
    flash('评论发表成功!')
    return redirect(url_for('movie_detail', movie_id=movie_id))


@app.route('/recommendations')
@login_required
def recommendations():
    user_id = session['user_id']
    recs, _ = get_recommendations(user_id=user_id, top_k=50)
    return render_template('recommendations.html', recommendations=recs)


# ==================== 管理员路由 ====================

@app.route('/admin')
@admin_required
def admin():
    db = get_db()
    user_count = db.execute('SELECT COUNT(*) as cnt FROM users').fetchone()['cnt']
    movie_count = db.execute('SELECT COUNT(*) as cnt FROM movies').fetchone()['cnt']
    rating_count = db.execute('SELECT COUNT(*) as cnt FROM ratings').fetchone()['cnt']
    review_count = db.execute('SELECT COUNT(*) as cnt FROM reviews').fetchone()['cnt']
    return render_template('admin.html',
                           user_count=user_count,
                           movie_count=movie_count,
                           rating_count=rating_count,
                           review_count=review_count)


@app.route('/admin/users')
@admin_required
def admin_users():
    db = get_db()
    users = db.execute('SELECT * FROM users ORDER BY id').fetchall()
    return render_template('admin_users.html', users=users)


@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    db = get_db()
    db.execute('DELETE FROM ratings WHERE user_id = ?', (user_id,))
    db.execute('DELETE FROM reviews WHERE user_id = ?', (user_id,))
    db.execute('DELETE FROM users WHERE id = ?', (user_id,))
    db.commit()
    flash('用户已删除')
    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/role', methods=['POST'])
@admin_required
def admin_change_role(user_id):
    new_role = request.form.get('role', 'user')
    db = get_db()
    db.execute('UPDATE users SET role = ? WHERE id = ?', (new_role, user_id))
    db.commit()
    flash('角色已更新')
    return redirect(url_for('admin_users'))


@app.route('/admin/movies')
@admin_required
def admin_movies():
    db = get_db()
    movies = db.execute('SELECT * FROM movies ORDER BY movie_id').fetchall()
    return render_template('admin_movies.html', movies=movies)


@app.route('/admin/movies/<int:movie_id>/delete', methods=['POST'])
@admin_required
def admin_delete_movie(movie_id):
    db = get_db()
    db.execute('DELETE FROM ratings WHERE movie_id = ?', (movie_id,))
    db.execute('DELETE FROM reviews WHERE movie_id = ?', (movie_id,))
    db.commit()
    flash('电影评分/评论已清理')
    return redirect(url_for('admin_movies'))


@app.route('/admin/reviews')
@admin_required
def admin_reviews():
    db = get_db()
    reviews = db.execute(
        'SELECT r.*, u.username, m.title FROM reviews r '
        'JOIN users u ON r.user_id = u.id '
        'JOIN movies m ON r.movie_id = m.movie_id '
        'ORDER BY r.created_at DESC LIMIT 100'
    ).fetchall()
    return render_template('admin_reviews.html', reviews=reviews)


@app.route('/admin/reviews/<int:review_id>/delete', methods=['POST'])
@admin_required
def admin_delete_review(review_id):
    db = get_db()
    db.execute('DELETE FROM reviews WHERE id = ?', (review_id,))
    db.commit()
    flash('评论已删除')
    return redirect(url_for('admin_reviews'))


# ==================== 启动 ====================

def run_app(host='127.0.0.1', port=5000, debug=True):
    print('Initializing database...')
    init_db()
    print('Loading model...')
    load_model()
    print('Loading movie metadata...')
    load_movie_data()
    # 在应用上下文中同步电影数据
    with app.app_context():
        populate_movies_table()
    print(f'Starting server at http://{host}:{port}')
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_app()
