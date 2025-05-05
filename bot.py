import os
import io
import pickle
from threading import Thread

from flask import Flask
import pandas as pd
import matplotlib.pyplot as plt
import openai
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Health-check web server for Render
health_app = Flask(__name__)

@health_app.route("/")
def health():
    return "OK"

def run_health_server():
    port = int(os.environ.get("PORT", 5000))
    health_app.run(host="0.0.0.0", port=port)

# 1. Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
openai.api_key = os.getenv("OPENAI_API_KEY")

# 2. Load RAG artifacts
with open("faiss_index.pkl", "rb") as f:
    index, docs = pickle.load(f)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Load Llama (via Hugging Face) for free-form generation

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)



# 4. Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hello! I'm CompostThinkers bot.\nUse /help to see commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start — Welcome message\n"
        "/dashboard — Send a sample chart\n"
        "/table — Send a sample data table\n"
        "/gen_image <prompt> — Generate an AI image\n"
        "/data_menu — Interactive menu of actions\n"
        "/ask <question> — Ask the bot using your custom data"
    )

async def send_dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = pd.DataFrame({'x': range(10), 'y': [i**2 + 5 for i in range(10)]})
    fig, ax = plt.subplots()
    ax.plot(df['x'], df['y'], marker='o')
    ax.set(title="Sample Quadratic", xlabel="x", ylabel="y")
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    await update.message.reply_photo(photo=buf, caption="📊 Here’s your dashboard!")

async def send_table(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = pd.DataFrame({"Name": ["Alice","Bob","Carol"], "Score": [82, 91, 78]})
    md = "```\n" + df.to_markdown(index=False) + "\n```"
    await update.message.reply_text(md, parse_mode="Markdown")

async def handle_image_gen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args) or "A serene mountain lake at sunrise"
    resp = openai.Image.create(prompt=prompt, n=1, size="512x512")
    url = resp['data'][0]['url']
    await update.message.reply_photo(photo=url, caption=f"🎨 {prompt}")

async def data_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    buttons = [
        [InlineKeyboardButton("Show Dashboard", callback_data="dash")],
        [InlineKeyboardButton("Show Table",     callback_data="table")],
        [InlineKeyboardButton("Gen Image",      callback_data="image")]
    ]
    await update.message.reply_text(
        "Choose an action:",
        reply_markup=InlineKeyboardMarkup(buttons)
    )

async def menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if q.data == "dash":
        await send_dashboard(update, context)
    elif q.data == "table":
        await send_table(update, context)
    else:
        await handle_image_gen(update, context)

async def ask_with_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.partition(' ')[2]
    if not query:
        await update.message.reply_text("Please ask a question after /ask, e.g. `/ask What is X?`")
        return
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k=3)
    context_chunks = [docs[i] for i in I[0]]
    prompt = (
        "You are a helpful assistant. Use the following data to answer the user:\n\n"
        + "\n\n".join(f"DATA[{i+1}]: {chunk}" for i, chunk in enumerate(context_chunks))
        + f"\n\nUser: {query}\nAssistant:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    await update.message.reply_text(answer)

# 5. Register handlers and start the bot

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("dashboard", send_dashboard))
    app.add_handler(CommandHandler("table", send_table))
    app.add_handler(CommandHandler("gen_image", handle_image_gen))
    app.add_handler(CommandHandler("data_menu", data_menu))
    app.add_handler(CallbackQueryHandler(menu_handler))
    app.add_handler(CommandHandler("ask", ask_with_data))

    app.run_polling()

if __name__ == "__main__":
    # start the health server for Render
    Thread(target=run_health_server).start()
    main()
