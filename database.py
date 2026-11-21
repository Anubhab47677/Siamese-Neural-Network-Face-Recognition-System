"""
database.py
-----------
SQLite storage of registered users and their 128-d face embeddings.
Images of a user are stored in <db folder>/<user_id>/
"""
import os, sqlite3, shutil, datetime
import numpy as np
from utils import DB_PATH, EMB_DIM


def _connect(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path=DB_PATH):
    with _connect(db_path) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users(
                        user_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT, n_images INTEGER)""")
        c.execute("""CREATE TABLE IF NOT EXISTS embeddings(
                        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, embedding BLOB NOT NULL,
                        FOREIGN KEY(user_id) REFERENCES users(user_id))""")


def user_exists(user_id, db_path=DB_PATH):
    init_db(db_path)
    with _connect(db_path) as c:
        return c.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone() is not None


def add_user(user_id, name, embeddings, n_images=None, db_path=DB_PATH, overwrite=False):
    """Create a user record (and store every embedding)."""
    init_db(db_path)
    embeddings = np.asarray(embeddings, dtype="float32").reshape(-1, EMB_DIM)
    if user_exists(user_id, db_path):
        if not overwrite:
            raise ValueError(f"User ID '{user_id}' already exists.")
        delete_user(user_id, db_path)
    with _connect(db_path) as c:
        c.execute("INSERT INTO users VALUES (?,?,?,?)",
                  (user_id, name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), n_images or len(embeddings)))
        c.executemany("INSERT INTO embeddings(user_id, embedding) VALUES (?,?)",
                      [(user_id, e.tobytes()) for e in embeddings])


def update_user_name(user_id, new_name, db_path=DB_PATH):
    init_db(db_path)
    with _connect(db_path) as c:
        c.execute("UPDATE users SET name=? WHERE user_id=?", (new_name, user_id))


def get_all_users(db_path=DB_PATH):
    init_db(db_path)
    with _connect(db_path) as c:
        rows = c.execute("SELECT user_id, name, created_at, n_images FROM users ORDER BY created_at").fetchall()
    return [dict(user_id=r[0], name=r[1], created_at=r[2], n_images=r[3]) for r in rows]


def get_all_embeddings(db_path=DB_PATH):
    """{user_id: {'name': str, 'embeddings': (n,128) array}}"""
    init_db(db_path)
    out = {u["user_id"]: {"name": u["name"], "embeddings": []} for u in get_all_users(db_path)}
    with _connect(db_path) as c:
        for uid, blob in c.execute("SELECT user_id, embedding FROM embeddings"):
            if uid in out:
                out[uid]["embeddings"].append(np.frombuffer(blob, dtype="float32"))
    for uid in out:
        out[uid]["embeddings"] = np.stack(out[uid]["embeddings"]) if out[uid]["embeddings"] else np.zeros((0, EMB_DIM), "float32")
    return out


def delete_user(user_id, db_path=DB_PATH):
    init_db(db_path)
    with _connect(db_path) as c:
        c.execute("DELETE FROM embeddings WHERE user_id=?", (user_id,))
        n = c.execute("DELETE FROM users WHERE user_id=?", (user_id,)).rowcount
    shutil.rmtree(os.path.join(os.path.dirname(db_path), user_id), ignore_errors=True)
    return n > 0
