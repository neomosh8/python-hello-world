#!/usr/bin/env python
# pyright: reportUnusedVariable=false, reportGeneralTypeIssues=false
"""
ربات تلگرام تخمین و تحویل پروژه پژوهشی (نسخه ریپلیت)

Hit RUN to execute the program.

این ربات به‌صورت تعاملی نوع پژوهش، نیازهای داده، نام و شماره تماس کاربر را دریافت می‌کند و سپس
هزینه و زمان انجام پروژه را برآورد می‌نماید. پس از تأیید کاربر، آی‌دی پژوهشگر برای هماهنگی
ارسال می‌شود.

Read the README.md file for more information on how to get and deploy Telegram bots.
"""

import logging
import os

from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ForceReply,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    ConversationHandler,
    CallbackQueryHandler,
)

# Bot token from environment variable
my_bot_token = os.environ.get('YOUR_BOT_TOKEN', '7822723984:AAGlyToZeJd5kDaULIhYwHNa-DNu9YePJuU')

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# -----------------------------
# پیکربندی ثابت
# -----------------------------
RESEARCHER_ID = "@Mohammadrzsx"  # آی‌دی تلگرامی پژوهشگر

# ضرایب پایه هزینه و زمان
BASE_COST = 10_000_000  # ریال
BASE_DAYS = 14  # روز کاری

# وضعیت‌های گفت‌وگو
(
    ASK_RESEARCH,
    ASK_DATA,
    ASK_PHONE,
    ASK_NAME,
    SHOW_ESTIMATE,
) = range(5)

# گزینه‌های پژوهش و داده
RESEARCH_OPTIONS = [
    ("پژوهش نظری", 1.0),
    ("پژوهش تجربی", 1.2),
    ("نگارش مقاله", 1.5),
    ("پروپوزال گرنت", 2.0),
]

DATA_OPTIONS = [
    ("بدون داده", 1.0),
    ("تحلیل دادهٔ موجود", 1.3),
    ("تولید دادهٔ مصنوعی", 1.6),
    ("جمع‌آوری میدانی", 2.0),
]

# -----------------------------
# User data storage (since we can't use persistent storage)
# -----------------------------
user_sessions = {}


def get_user_data(user_id):
    if user_id not in user_sessions:
        user_sessions[user_id] = {}
    return user_sessions[user_id]


def clear_user_data(user_id):
    if user_id in user_sessions:
        del user_sessions[user_id]


# -----------------------------
# Command handlers
# -----------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """شروع فرایند با انتخاب نوع پژوهش"""
    user = update.effective_user
    user_data = get_user_data(user.id)

    await update.message.reply_html(
        f"سلام {user.mention_html()}!\n"
        "به ربات تخمین پروژه پژوهشی خوش آمدید.\n\n"
        "برای شروع، نوع پژوهش خود را انتخاب کنید:",
        reply_markup=ForceReply(selective=True),
    )

    kb = [[text] for text, _ in RESEARCH_OPTIONS]
    reply_markup = ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=True)

    await update.message.reply_text(
        "نوع پژوهش را انتخاب کنید:",
        reply_markup=reply_markup
    )
    return ASK_RESEARCH


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """نمایش راهنما"""
    help_text = (
        "🤖 راهنمای ربات تخمین پروژه پژوهشی\n\n"
        "این ربات به شما کمک می‌کند تا:\n"
        "🔹 هزینه پروژه پژوهشی خود را تخمین بزنید\n"
        "🔹 زمان تحویل را محاسبه کنید\n"
        "🔹 با پژوهشگر ارتباط برقرار کنید\n\n"
        "دستورات:\n"
        "/start - شروع فرایند تخمین\n"
        "/help - نمایش این راهنما\n"
        "/cancel - لغو فرایند جاری\n\n"
        "برای شروع /start را بفرستید."
    )
    await update.message.reply_text(help_text)


async def research_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """انتخاب نوع پژوهش"""
    user_data = get_user_data(update.effective_user.id)
    choice = update.message.text

    # بررسی انتخاب معتبر
    found = False
    for text, factor in RESEARCH_OPTIONS:
        if choice == text:
            user_data["research_text"] = text
            user_data["research_factor"] = factor
            found = True
            break

    if not found:
        kb = [[text] for text, _ in RESEARCH_OPTIONS]
        reply_markup = ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=True)
        await update.message.reply_text(
            "لطفاً از گزینه‌های موجود انتخاب کنید:",
            reply_markup=reply_markup
        )
        return ASK_RESEARCH

    # انتقال به مرحله بعدی
    kb = [[text] for text, _ in DATA_OPTIONS]
    reply_markup = ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=True)

    await update.message.reply_text(
        "مربوط به داده چه گزینه‌ای مناسب است؟",
        reply_markup=reply_markup
    )
    return ASK_DATA


async def data_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """انتخاب نوع داده"""
    user_data = get_user_data(update.effective_user.id)
    choice = update.message.text

    # بررسی انتخاب معتبر
    found = False
    for text, factor in DATA_OPTIONS:
        if choice == text:
            user_data["data_text"] = text
            user_data["data_factor"] = factor
            found = True
            break

    if not found:
        kb = [[text] for text, _ in DATA_OPTIONS]
        reply_markup = ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=True)
        await update.message.reply_text(
            "لطفاً از گزینه‌های موجود انتخاب کنید:",
            reply_markup=reply_markup
        )
        return ASK_DATA

    # درخواست شماره تماس
    contact_btn = KeyboardButton("📱 ارسال شماره تماس", request_contact=True)
    skip_btn = KeyboardButton("رد کردن")
    reply_markup = ReplyKeyboardMarkup(
        [[contact_btn], [skip_btn]],
        resize_keyboard=True,
        one_time_keyboard=True
    )

    await update.message.reply_text(
        "لطفاً شماره تماس خود را ارسال کنید یا رد کنید:",
        reply_markup=reply_markup
    )
    return ASK_PHONE


async def phone_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """دریافت شماره تماس"""
    user_data = get_user_data(update.effective_user.id)

    if update.message.contact:
        phone = update.message.contact.phone_number
        user_data["phone"] = phone
    elif update.message.text == "رد کردن":
        user_data["phone"] = "ارائه نشده"
    else:
        phone = update.message.text.strip()
        user_data["phone"] = phone

    await update.message.reply_text(
        "نام کامل خود را وارد کنید:",
        reply_markup=ReplyKeyboardMarkup([["رد کردن"]], resize_keyboard=True, one_time_keyboard=True)
    )
    return ASK_NAME


async def name_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """دریافت نام و محاسبه تخمین"""
    user_data = get_user_data(update.effective_user.id)

    if update.message.text == "رد کردن":
        name = "ارائه نشده"
    else:
        name = update.message.text.strip()

    user_data["name"] = name

    # محاسبه هزینه و زمان
    research_factor = user_data.get("research_factor", 1.0)
    data_factor = user_data.get("data_factor", 1.0)

    cost = int(BASE_COST * research_factor * data_factor)
    days = int(BASE_DAYS * research_factor * data_factor)

    fast_cost = int(cost * 1.25)
    fast_days = max(days - 5, 3)

    user_data.update({
        "cost": cost,
        "days": days,
        "fast_cost": fast_cost,
        "fast_days": fast_days,
    })

    # نمایش تخمین
    estimate_text = (
        f"📊 تخمین پروژه شما:\n\n"
        f"🔹 نوع پژوهش: {user_data['research_text']}\n"
        f"🔹 نوع داده: {user_data['data_text']}\n"
        f"🔹 نام: {name}\n"
        f"🔹 تماس: {user_data['phone']}\n\n"
        f"💰 برآورد هزینه: {cost:,} ریال\n"
        f"⏰ زمان تحویل: {days} روز کاری\n\n"
        f"⚡ گزینه تسریع (+۲۵٪):\n"
        f"💰 هزینه: {fast_cost:,} ریال\n"
        f"⏰ زمان: {fast_days} روز کاری"
    )

    buttons = [
        [
            InlineKeyboardButton("✅ تایید معمولی", callback_data="APPROVE_NORMAL"),
            InlineKeyboardButton("⚡ تایید تسریع", callback_data="APPROVE_FAST"),
        ],
        [InlineKeyboardButton("❌ لغو", callback_data="CANCEL")],
    ]

    await update.message.reply_text(
        estimate_text,
        reply_markup=InlineKeyboardMarkup(buttons)
    )
    return SHOW_ESTIMATE


async def approve_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """پردازش تایید یا لغو پروژه"""
    query = update.callback_query
    user_data = get_user_data(update.effective_user.id)

    await query.answer()
    data = query.data

    if data == "CANCEL":
        await query.edit_message_text(
            "❌ فرایند لغو شد.\n\n"
            "برای شروع دوباره /start را بزنید."
        )
        clear_user_data(update.effective_user.id)
        return ConversationHandler.END

    if data == "APPROVE_NORMAL":
        final_cost = user_data["cost"]
        final_days = user_data["days"]
        service_type = "معمولی"
    else:  # APPROVE_FAST
        final_cost = user_data["fast_cost"]
        final_days = user_data["fast_days"]
        service_type = "تسریع"

    await query.edit_message_text(
        f"✅ پروژه تأیید شد!\n\n"
        f"🔹 نوع سرویس: {service_type}\n"
        f"💰 هزینه نهایی: {final_cost:,} ریال\n"
        f"⏰ زمان تحویل: {final_days} روز کاری\n\n"
        f"📝 اطلاعات شما ثبت شد و به زودی با شما تماس خواهیم گرفت."
    )

    await query.message.reply_text(
        f"🔗 برای هماهنگی بیشتر:\n"
        f"👤 پژوهشگر: {RESEARCHER_ID}\n\n"
        f"📞 لطفاً در تماس باشید تا مراحل بعدی را هماهنگ کنیم."
    )

    clear_user_data(update.effective_user.id)
    return ConversationHandler.END


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """لغو فرایند"""
    await update.message.reply_text(
        "❌ فرایند لغو شد.\n\n"
        "برای شروع دوباره /start را بزنید."
    )
    clear_user_data(update.effective_user.id)
    return ConversationHandler.END


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(my_bot_token).build()

    # Setup conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASK_RESEARCH: [MessageHandler(filters.TEXT & ~filters.COMMAND, research_choice)],
            ASK_DATA: [MessageHandler(filters.TEXT & ~filters.COMMAND, data_choice)],
            ASK_PHONE: [
                MessageHandler(filters.CONTACT, phone_received),
                MessageHandler(filters.TEXT & ~filters.COMMAND, phone_received),
            ],
            ASK_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, name_received)],
            SHOW_ESTIMATE: [CallbackQueryHandler(approve_callback)],
        },
        fallbacks=[CommandHandler("cancel", cancel_command)],
        allow_reentry=True,
    )

    # Add handlers
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()