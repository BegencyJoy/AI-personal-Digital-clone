

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch, imaplib, email, smtplib, time, datetime, os, random
from email.mime.text import MIMEText


# Model & Tokenizer

MODEL_NAME = r"D:\python2.0\AI-personal-DIgital_Twin\digital_twin_lora_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()


# Personality Dataset

DATA_FILE = r"D:\python2.0\AI-personal-DIgital_Twin\converted_dataset.jsonl"
dataset = load_dataset("json", data_files=DATA_FILE)["train"]


# Improved Few-Shot Prompt Builder

def build_personality_prompt(dataset, max_chars=2000):
    """Select diverse examples up to a safe character limit for stable tone anchoring."""
    examples = random.sample(list(dataset), min(25, len(dataset)))
    text, total_len = "", 0
    for ex in examples:
        example = f"Q: {ex['prompt']}\nA: {ex['response']}\n\n"
        if total_len + len(example) > max_chars:
            break
        text += example
        total_len += len(example)
    return f"""You are Begency's digital twin.
You reply naturally, confidently, and in Begency’s phrasing.
Examples of my tone and responses:
{text}
"""


# Improved Text Generator

def generate_text(user_input, personality_prompt, max_length=300):
    """
    Generates a more accurate, human-like response using only the fine-tuned local model.
    Anchors the model with few-shot personality examples and balanced decoding parameters.
    """
    if not user_input.strip():
        return "I'm here — could you please clarify what you meant?"

    # Build prompt with personality tone
    prompt = f"""
You are Begency's AI clone. Reply naturally and personally like Begency would.
Base your response on the examples below:

{personality_prompt}

Now respond to this new message:
User: {user_input}
Me:
"""

    # Tokenize input safely
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768,  # stay within model context window
    )

    # Generate with controlled creativity and clarity
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,  # replaces max_length math
            temperature=0.55,            # lower = more focused, accurate
            top_p=0.85,                  # moderate sampling diversity
            repetition_penalty=1.2,      # avoids looping or generic phrases
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the model's new output (skip prompt)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Clean up
    response = response.replace("User:", "").replace("Me:", "").strip()

    # Always end with your signature line
    if not response.lower().endswith("\nthis is ai reply."):
        response += "\n\n\nRegards\nBegency E."

    return response




# Email Configuration

EMAIL_ADDRESS = "begencyjoy@gmail.com"
EMAIL_PASSWORD = "wer"  # App password
IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Only reply to specific senders
allowed_senders = ["begencyjoy16@gmail.com"]

# Logging
log_file = "replied_log.txt"
if not os.path.exists(log_file):
    open(log_file, "w").close()

def has_replied(message_id):
    with open(log_file, "r") as f:
        return message_id in f.read().splitlines()

def mark_as_replied(message_id):
    with open(log_file, "a") as f:
        f.write(message_id + "\n")


# Check & Reply Function

def check_and_reply_emails():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("inbox")

        today = datetime.date.today().strftime("%d-%b-%Y")
        status, messages = mail.search(None, f'(UNSEEN SINCE {today})')
        email_ids = messages[0].split()

        if not email_ids:
            print("No new emails today.")
            mail.logout()
            return

        for e_id in email_ids:
            status, msg_data = mail.fetch(e_id, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])
            sender = msg["From"]
            subject = msg["Subject"] or "(No Subject)"
            message_id = msg["Message-ID"] or e_id.decode()

            if has_replied(message_id):
                continue

            # Check allowed sender
            if not any(allowed in sender for allowed in allowed_senders):
                print(f"Ignoring sender: {sender}")
                continue

            # Get email body
            if msg.is_multipart():
                body = ""
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode(errors="ignore")
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")

            body = body.strip()[:800]  # limit input length

            print(f"\n From: {sender}\nSubject: {subject}")

            # Build prompt
            personality_prompt = build_personality_prompt(dataset)

            # Generate reply
            reply_text = generate_text(body, personality_prompt)
            reply_text += "\n\nThis is an AI clone reply."

            print(f" Reply:\n{reply_text}\n")

            # Send reply
            msg_to_send = MIMEText(reply_text)
            msg_to_send["Subject"] = "Re: " + subject
            msg_to_send["From"] = EMAIL_ADDRESS
            msg_to_send["To"] = sender

            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, sender, msg_to_send.as_string())
            server.quit()

            mark_as_replied(message_id)
            print(" Reply sent successfully.")

        mail.logout()

    except Exception as e:
        print(" Error:", e)


# Loop

print(" Digital Twin Email Clone running... (checks every 1 minute)")
while True:
    check_and_reply_emails()
    time.sleep(60)
