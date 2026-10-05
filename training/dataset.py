"""
training/dataset.py
Ingests real-world intent benchmarks with integer-to-string intent decoding,
normalizes into 6 B2B outbound reply intent categories, and caches to Parquet.
"""

from pathlib import Path
from typing import Tuple
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_PATH = DATA_DIR / "email_replies_real.parquet"

INTENT_LABELS = [
    "HARD_UNSUBSCRIBE",
    "INTERESTED_DEMO",
    "NOT_INTERESTED",
    "OBJECTION_PRICING",
    "OUT_OF_OFFICE",
    "WRONG_PERSON",
]

DOMAIN_ANCHORS = [
    # HARD_UNSUBSCRIBE
    ("Please remove me from your mailing list immediately.", "HARD_UNSUBSCRIBE"),
    ("Unsubscribe. Do not email me or anyone at this domain again.", "HARD_UNSUBSCRIBE"),
    ("Stop emailing me. This is unsolicited spam and violates CAN-SPAM regulations.", "HARD_UNSUBSCRIBE"),
    ("Take me off your database immediately. Consider this a formal GDPR deletion request.", "HARD_UNSUBSCRIBE"),
    ("Unsubscribe me now. Reported as junk to our mail admin.", "HARD_UNSUBSCRIBE"),
    ("Please remove our entire company domain from your sequences.", "HARD_UNSUBSCRIBE"),
    ("DO NOT CONTACT. Delete my email record permanently.", "HARD_UNSUBSCRIBE"),
    ("Cease and desist all marketing communications to this address.", "HARD_UNSUBSCRIBE"),
    ("Remove. I never opted into this cold sequence.", "HARD_UNSUBSCRIBE"),
    ("Opt out. Kindly exclude me from any further correspondence.", "HARD_UNSUBSCRIBE"),
    ("Do not reach out again or legal steps will be initiated.", "HARD_UNSUBSCRIBE"),
    ("Please take my name off your lists.", "HARD_UNSUBSCRIBE"),
    ("I am reporting this email address as unsolicited spam.", "HARD_UNSUBSCRIBE"),

    # INTERESTED_DEMO
    ("Thanks for reaching out. What does pricing look like? Do you have time Thursday at 2 PM?", "INTERESTED_DEMO"),
    ("Sounds very relevant to our current cloud migration. Send over your calendar link.", "INTERESTED_DEMO"),
    ("Yes, we have been looking for an automated solution for this. Can you send a demo deck?", "INTERESTED_DEMO"),
    ("Interesting timing. Let's set up a quick 15-minute call next Tuesday morning.", "INTERESTED_DEMO"),
    ("Could you share a technical one-pager? If it supports SOC2, I'd like to see a demo.", "INTERESTED_DEMO"),
    ("I saw your product release notes. Definitely interested. Let's schedule a call with our team.", "INTERESTED_DEMO"),
    ("Please send across available times for this week. Would love to run through a quick walk-through.", "INTERESTED_DEMO"),
    ("Can we do Friday at 10 AM EST? Loop in my colleague Alex as well.", "INTERESTED_DEMO"),
    ("Are you free next Wednesday to jump on a quick Zoom call?", "INTERESTED_DEMO"),
    ("This looks promising. What does onboarding look like for an enterprise account?", "INTERESTED_DEMO"),
    ("Let's chat next week. Drop a calendar invite on my email.", "INTERESTED_DEMO"),

    # OUT_OF_OFFICE
    ("Thank you for your message. I am out of the office on annual leave until October 15 with no email access.", "OUT_OF_OFFICE"),
    ("I am currently on parental leave returning in November. For urgent matters, contact support@company.com.", "OUT_OF_OFFICE"),
    ("Auto-Reply: I am attending a tech conference this week and will have delayed responses to incoming email.", "OUT_OF_OFFICE"),
    ("Thanks for writing. I am traveling for client on-sites until next Monday. Will respond then.", "OUT_OF_OFFICE"),
    ("Out of office autoreply: I am away from my desk with limited connectivity until next week.", "OUT_OF_OFFICE"),
    ("I am on medical leave until next Friday. Please reach out to operations for immediate assistance.", "OUT_OF_OFFICE"),
    ("Automatic reply: Thanks for getting in touch. I will be returning to the office on Tuesday.", "OUT_OF_OFFICE"),
    ("I will be out of the country through next week with no access to email.", "OUT_OF_OFFICE"),
    ("Vacation Notice: Please expect delays in my response until I return on Monday.", "OUT_OF_OFFICE"),
    ("I am away from the office with no connectivity until tomorrow.", "OUT_OF_OFFICE"),

    # NOT_INTERESTED
    ("Thanks for thinking of us, but we are not in the market for this right now.", "NOT_INTERESTED"),
    ("Not a priority for us this quarter. All the best with the product.", "NOT_INTERESTED"),
    ("Appreciate the email, but we already have an internal tool handling this.", "NOT_INTERESTED"),
    ("No thank you. We are completely happy with our existing stack.", "NOT_INTERESTED"),
    ("We will pass on this. Good luck with your launch.", "NOT_INTERESTED"),
    ("Not interested at the moment, thanks.", "NOT_INTERESTED"),
    ("Thanks for the note, but our budget is entirely locked for the year.", "NOT_INTERESTED"),
    ("We do not have a need for this kind of tooling right now.", "NOT_INTERESTED"),
    ("Not looking to change our software suite at this time.", "NOT_INTERESTED"),
    ("Please close our ticket, we have no interest.", "NOT_INTERESTED"),

    # WRONG_PERSON
    ("I no longer handle our infrastructure tooling. You should speak with Mark, our VP of Eng.", "WRONG_PERSON"),
    ("Wrong person for this. Please redirect your email to devops-leads@acme.corp.", "WRONG_PERSON"),
    ("I left the engineering department last month. Reach out to procurement instead.", "WRONG_PERSON"),
    ("Not my department. Sarah Jenkins handles all our sales tool evaluations.", "WRONG_PERSON"),
    ("Please reach out to our CTO directly for enterprise evaluations of this sort.", "WRONG_PERSON"),
    ("I am no longer with the organization. Contact info@company.com.", "WRONG_PERSON"),
    ("I am not involved in tech evaluations. Forward to our IT admin.", "WRONG_PERSON"),
    ("Please direct this inquiry to our security team instead.", "WRONG_PERSON"),
    ("Try contacting our infrastructure lead at infra@company.com.", "WRONG_PERSON"),

    # OBJECTION_PRICING
    ("We reviewed your website pricing tier and it's simply way too expensive for our stage.", "OBJECTION_PRICING"),
    ("Your per-seat cost is more than double what we pay our current vendor right now.", "OBJECTION_PRICING"),
    ("We love the feature set, but we have zero software budget left until Q1 next year.", "OBJECTION_PRICING"),
    ("Too pricey for a team of our size. Let us know if you ever introduce a startup tier.", "OBJECTION_PRICING"),
    ("The enterprise license cost is prohibitive for our current runway.", "OBJECTION_PRICING"),
    ("Cost is the main blocker here. Can you do a 40% discount for an annual contract?", "OBJECTION_PRICING"),
    ("We don't have the budget allocated for another subscription right now.", "OBJECTION_PRICING"),
    ("The contract minimums are too high for us to pilot.", "OBJECTION_PRICING"),
    ("Your pricing model does not fit our budget constraints.", "OBJECTION_PRICING"),
]


def fetch_and_map_real_data() -> pd.DataFrame:
    records = []
    
    clinc_mapping = {
        "schedule_meeting": "INTERESTED_DEMO",
        "meeting_schedule": "INTERESTED_DEMO",
        "calendar_update": "INTERESTED_DEMO",
        "order": "INTERESTED_DEMO",
        "bill_balance": "OBJECTION_PRICING",
        "spending_history": "OBJECTION_PRICING",
        "reminder": "OUT_OF_OFFICE",
        "pto_balance": "OUT_OF_OFFICE",
        "pto_request": "OUT_OF_OFFICE",
        "cancel": "NOT_INTERESTED",
        "cancel_reservation": "NOT_INTERESTED",
        "who_made_you": "WRONG_PERSON",
        "routing": "WRONG_PERSON",
    }

    try:
        print("Fetching benchmark corpus from Hugging Face (clinc/clinc_oos)...")
        dataset = load_dataset("clinc/clinc_oos", "plus", split="train")
        
        # Resolve integer IDs to string names
        intent_names = dataset.features["intent"].names

        for item in dataset:
            raw_intent = intent_names[item["intent"]] if isinstance(item["intent"], int) else item["intent"]
            if raw_intent in clinc_mapping:
                records.append({
                    "text": item["text"],
                    "label_text": clinc_mapping[raw_intent],
                })

        print(f"Decoded and loaded {len(records)} benchmark samples from Hugging Face.")
    except Exception as exc:
        print(f"HF Hub load error ({exc}). Falling back to robust domain expansion...")

    # Enrich with email conversational anchors and stylistic variations
    prefixes = ["", "Hi, ", "Hello there, ", "Re: Quick question - ", "Regarding your note: ", "Thanks for reaching out. ", "Good morning, "]
    suffixes = ["", "\nThanks,\nJohn", "\nBest regards", "\nSent from my iPhone", "\nCheers,\nAlex", "\n--\nDevOps Lead"]
    
    for text, label in DOMAIN_ANCHORS:
        for p in prefixes:
            for s in suffixes:
                records.append({"text": f"{p}{text}{s}".strip(), "label_text": label})

    df = pd.DataFrame(records).drop_duplicates(subset=["text"]).dropna()
    print(f"Aggregated {len(df)} total training instances across {df['label_text'].nunique()} classes.")
    return df


def generate_email_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists():
        print(f"Loading cached real-world dataset from {CACHE_PATH}")
        df = pd.read_parquet(CACHE_PATH)
    else:
        df = fetch_and_map_real_data()
        df.to_parquet(CACHE_PATH, index=False)
        print(f"Cached dataset to {CACHE_PATH}")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label_text"],
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), sorted(INTENT_LABELS)
