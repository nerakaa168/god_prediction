import os
from collections import deque
from dataclasses import dataclass
from dotenv import load_dotenv
import telebot
from telebot import types

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("BOT_TOKEN belum diisi di .env")

bot = telebot.TeleBot(TOKEN, parse_mode="HTML")

MAX_HISTORY = 300
MIN_FOR_PRED = 10
PRED_WINDOW = 15  # ambil last 15 untuk prediksi (kalau data >=15)

@dataclass
class UserState:
    history: deque
    last_added: str | None = None

USER: dict[int, UserState] = {}

def get_state(chat_id: int) -> UserState:
    if chat_id not in USER:
        USER[chat_id] = UserState(history=deque(maxlen=MAX_HISTORY))
    return USER[chat_id]

def normalize_result(text: str) -> str | None:
    t = text.strip().upper()
    if t in ("P", "PLAYER"):
        return "P"
    if t in ("B", "BANKER"):
        return "B"
    if t in ("T", "TIE"):
        return "T"
    return None

def make_keyboard():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.row("➕ P", "➕ B", "➕ T")
    kb.row("▶️ Next Prediction", "🧹 Reset")
    kb.row("📊 Stats", "↩️ Undo", "❓ Help")
    return kb

def streak(history: deque) -> tuple[str | None, int]:
    if not history:
        return None, 0
    last = history[-1]
    s = 1
    for i in range(len(history) - 2, -1, -1):
        if history[i] == last:
            s += 1
        else:
            break
    return last, s

def dirichlet_posterior_mean(counts: dict[str, int], alpha=None) -> dict[str, float]:
    # 3-kelas (P/B/T) smoothing
    if alpha is None:
        alpha = {"P": 1.0, "B": 1.0, "T": 1.0}
    total = sum(counts.values())
    denom = sum(alpha.values()) + total
    return {k: (alpha[k] + counts.get(k, 0)) / denom for k in alpha}

def counts_from(seq: list[str]) -> dict[str, int]:
    return {
        "P": sum(1 for x in seq if x == "P"),
        "B": sum(1 for x in seq if x == "B"),
        "T": sum(1 for x in seq if x == "T"),
    }

def send_stats(chat_id: int):
    st = get_state(chat_id)
    h = list(st.history)
    n = len(h)
    if n == 0:
        bot.send_message(chat_id, "Belum ada data. Input P/B/T dulu.", reply_markup=make_keyboard())
        return

    c = counts_from(h)
    pct = {k: (c[k] / n) * 100 for k in c}

    last, st_len = streak(st.history)
    name = {"P": "Player", "B": "Banker", "T": "Tie"}
    last_txt = name.get(last, "-")

    last20 = "".join(h[-20:])

    bot.send_message(
        chat_id,
        "<b>📊 Stats</b>\n"
        f"Total data: <b>{n}</b>\n"
        f"Player: <b>{c['P']}</b> ({pct['P']:.1f}%)\n"
        f"Banker: <b>{c['B']}</b> ({pct['B']:.1f}%)\n"
        f"Tie: <b>{c['T']}</b> ({pct['T']:.1f}%)\n\n"
        f"Streak terakhir: <b>{last_txt}</b> x <b>{st_len}</b>\n"
        f"20 terakhir: <code>{last20}</code>\n\n"
        f"Tekan <b>▶️ Next Prediction</b> kalau data sudah ≥{MIN_FOR_PRED}.",
        reply_markup=make_keyboard()
    )

def next_prediction(chat_id: int):
    st = get_state(chat_id)
    h = list(st.history)
    n = len(h)

    if n < MIN_FOR_PRED:
        bot.send_message(
            chat_id,
            f"Data belum cukup. Minimal <b>{MIN_FOR_PRED}</b> result dulu.\n"
            f"Sekarang baru: <b>{n}</b>",
            reply_markup=make_keyboard()
        )
        return

    window = h[-min(PRED_WINDOW, n):]  # last 15 atau kurang kalau data <15
    w_n = len(window)
    c = counts_from(window)

    # posterior (smoothing) – kamu bisa ubah prior kalau mau
    post = dirichlet_posterior_mean(c, alpha={"P": 1.0, "B": 1.0, "T": 1.0})

    name = {"P": "Player", "B": "Banker", "T": "Tie"}
    pick_key = max(post, key=post.get)
    pick_name = name[pick_key]

    last_seq = "".join(window)

    bot.send_message(
        chat_id,
        "<b>▶️ Next Prediction</b>\n"
        f"Basis: <b>last {w_n}</b> data\n"
        f"Sequence: <code>{last_seq}</code>\n\n"
        "<b>Estimasi peluang berikutnya (smoothing)</b>\n"
        f"P(Player) ≈ <b>{post['P']*100:.1f}%</b>\n"
        f"P(Banker) ≈ <b>{post['B']*100:.1f}%</b>\n"
        f"P(Tie) ≈ <b>{post['T']*100:.1f}%</b>\n\n"
        f"<b>Pick estimasi tertinggi:</b> <b>{pick_name}</b>\n"
        "<i>Catatan: ini estimasi dari window input, bukan jaminan hasil ronde berikutnya.</i>",
        reply_markup=make_keyboard()
    )

def do_undo(chat_id: int):
    st = get_state(chat_id)
    if not st.history:
        bot.send_message(chat_id, "Tidak ada yang bisa di-undo.", reply_markup=make_keyboard())
        return
    removed = st.history.pop()
    st.last_added = None
    bot.send_message(chat_id, f"Undo ✅ (hapus: <b>{removed}</b>)", reply_markup=make_keyboard())

def do_reset(chat_id: int):
    st = get_state(chat_id)
    st.history.clear()
    st.last_added = None
    bot.send_message(chat_id, "Reset ✅ Data history sudah dihapus.", reply_markup=make_keyboard())

@bot.message_handler(commands=["start"])
def start(m):
    bot.send_message(
        m.chat.id,
        "Input hasil ronde: <b>P</b>, <b>B</b>, atau <b>T</b>.\n"
        f"Kalau sudah isi 10–15 result, tekan <b>▶️ Next Prediction</b> untuk estimasi ronde berikutnya.\n\n"
        "Tombol tersedia di keyboard bawah.",
        reply_markup=make_keyboard()
    )

@bot.message_handler(commands=["stats"])
def stats_cmd(m): send_stats(m.chat.id)

@bot.message_handler(commands=["reset"])
def reset_cmd(m): do_reset(m.chat.id)

@bot.message_handler(commands=["undo"])
def undo_cmd(m): do_undo(m.chat.id)

@bot.message_handler(commands=["next"])
def next_cmd(m): next_prediction(m.chat.id)

@bot.message_handler(func=lambda m: True)
def handle(m):
    chat_id = m.chat.id
    text = (m.text or "").strip()

    # tombol cepat
    if text == "➕ P": text = "P"
    elif text == "➕ B": text = "B"
    elif text == "➕ T": text = "T"
    elif text == "🧹 Reset": return do_reset(chat_id)
    elif text == "▶️ Next Prediction": return next_prediction(chat_id)
    elif text == "📊 Stats": return send_stats(chat_id)
    elif text == "↩️ Undo": return do_undo(chat_id)
    elif text == "❓ Help":
        return bot.send_message(
            chat_id,
            "<b>Help</b>\n"
            "• Input: P / B / T\n"
            "• Setelah data ≥10, tekan ▶️ Next Prediction\n"
            "• Reset untuk hapus data\n",
            reply_markup=make_keyboard()
        )

    r = normalize_result(text)
    if not r:
        bot.send_message(chat_id, "Input tidak dikenali. Kirim <code>P</code>/<code>B</code>/<code>T</code>.", reply_markup=make_keyboard())
        return

    st = get_state(chat_id)
    st.history.append(r)
    st.last_added = r

    # cuma konfirmasi input (nggak auto-prediksi), biar sesuai maumu
    n = len(st.history)
    bot.send_message(
        chat_id,
        f"Masuk ✅ <b>{r}</b> (total: <b>{n}</b>)\n"
        f"Kalau sudah 10–15 data, tekan <b>▶️ Next Prediction</b>.",
        reply_markup=make_keyboard()
    )

if __name__ == "__main__":
    print("Bot jalan...")
    bot.infinity_polling(skip_pending=True)
