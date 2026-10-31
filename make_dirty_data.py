"""Generate a realistic dirty CSV (Indian e-commerce orders) plus an answer key.

Outputs (in ./dataset):
  orders_dirty.csv            - the messy input for the tool
  orders_ground_truth.csv     - the clean version (for measuring accuracy)
  planted_issues.csv          - every planted problem (the answer key)
  answer_key_summary.json     - counts per issue type / column
"""
import csv, json, random, copy, os
from datetime import date, timedelta
from collections import Counter

random.seed(42)
N_ORDERS, N_CUST = 500, 340
OUT = "dataset"
os.makedirs(OUT, exist_ok=True)

COLS = ["order_id", "customer_id", "customer_name", "email", "phone",
        "signup_date", "order_date", "city", "country", "product_category",
        "quantity", "unit_price", "total_amount", "payment_method",
        "status", "rating"]

FIRST = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Arjun", "Kavya",
         "Rahul", "Meera", "Siddharth", "Pooja", "Karan", "Neha", "Aditya", "Isha",
         "Manish", "Divya", "Nikhil", "Ritu", "Sanjay", "Tanvi", "Harsh", "Shreya",
         "Deepak", "Lakshmi", "Faisal", "Zoya", "Gurpreet", "Anjali"]
LAST = ["Sharma", "Patel", "Iyer", "Reddy", "Singh", "Mehta", "Nair", "Gupta",
        "Joshi", "Kulkarni", "Verma", "Das", "Khan", "Bose", "Chopra", "Menon",
        "Agarwal", "Desai", "Pillai", "Kapoor", "Banerjee", "Shetty", "Malhotra", "Rao"]
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "rediffmail.com"]
GEO = {"India": ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
                 "Pune", "Ahmedabad", "Jaipur", "Gurugram"],
       "United Arab Emirates": ["Dubai", "Abu Dhabi"],
       "Singapore": ["Singapore"],
       "United States": ["New York", "San Jose"]}
CODE = {"India": "+91", "United Arab Emirates": "+971", "Singapore": "+65", "United States": "+1"}
CAT = {"Electronics": (1500, 45000), "Clothing": (400, 4000), "Home & Kitchen": (300, 9000),
       "Books": (150, 1200), "Beauty": (200, 3000), "Sports": (500, 12000), "Grocery": (50, 1500)}
PAY = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash on Delivery"]
STATUS = ["Delivered", "Shipped", "Cancelled", "Returned", "Pending"]


def rand_date(a, b):
    return a + timedelta(days=random.randint(0, (b - a).days))


# ---------------------------------------------------------------- clean data
customers = []
for c in range(N_CUST):
    f, l = random.choice(FIRST), random.choice(LAST)
    country = random.choices(list(GEO), weights=[88, 4, 4, 4])[0]
    d = random.choice("6789") + "".join(random.choices("0123456789", k=9))
    code = CODE[country]
    customers.append(dict(
        id=f"CUST-{2000 + c}", name=f"{f} {l}",
        email=f"{f}.{l}{c}@{random.choice(DOMAINS)}".lower(),
        code=code, digits=d, phone=f"{code} {d[:5]} {d[5:]}",
        signup=rand_date(date(2023, 1, 1), date(2025, 6, 30)),
        city=random.choice(GEO[country]), country=country))

base = []
for o in range(N_ORDERS):
    cu = random.choice(customers)
    cat = random.choice(list(CAT))
    lo, hi = CAT[cat]
    price = max(10, random.randint(lo // 10, hi // 10) * 10)
    qty = random.choices([1, 2, 3, 4, 5], weights=[50, 25, 12, 8, 5])[0]
    od = rand_date(max(cu["signup"], date(2025, 1, 1)), date(2026, 9, 30))
    status = random.choices(STATUS, weights=[62, 12, 8, 8, 10])[0]
    rating = str(random.choices([1, 2, 3, 4, 5], weights=[5, 8, 15, 35, 37])[0]) \
        if status in ("Delivered", "Returned") else ""   # legitimately blank otherwise
    base.append(dict(
        _uid=o, order_id=f"ORD-{10001 + o}", customer_id=cu["id"],
        customer_name=cu["name"], email=cu["email"], phone=cu["phone"],
        signup_date=cu["signup"].isoformat(), order_date=od.isoformat(),
        city=cu["city"], country=cu["country"], product_category=cat,
        quantity=str(qty), unit_price=f"{price:.2f}", total_amount=f"{price * qty:.2f}",
        payment_method=random.choice(PAY), status=status, rating=rating))
orig = copy.deepcopy(base)
cust_by_id = {c["id"]: c for c in customers}

# ------------------------------------------------------------ corruptions
issues, used = [], set()


def pick(col, k, cond=lambda i: True):
    cand = [i for i in range(N_ORDERS) if (i, col) not in used and cond(i)]
    chosen = random.sample(cand, min(k, len(cand)))
    for i in chosen:
        used.add((i, col))
    return chosen


def log(i, col, typ, new, note=""):
    issues.append(dict(uid=orig[i]["_uid"], order_id=orig[i]["order_id"], column=col,
                       issue_type=typ, clean_value=orig[i][col], dirty_value=new, note=note))
    base[i][col] = new


# 1. missing values (blank + disguised placeholders)
PLACEHOLDERS = ["", "", "", "N/A", "null", "NA", "-", "none", "?", "NULL", "nan", "Unknown"]
for col, k in {"email": 25, "phone": 30, "city": 20, "unit_price": 12, "customer_name": 8,
               "quantity": 8, "order_date": 6, "rating": 20}.items():
    cond = (lambda i: orig[i]["rating"] != "") if col == "rating" else (lambda i: True)
    for i in pick(col, k, cond):
        ph = random.choice(PLACEHOLDERS)
        log(i, col, "missing_value", ph, "blank" if ph == "" else f"placeholder: {ph}")

# 2. impossible / future dates, logical date conflict
for i, bad in zip(pick("order_date", 4), ["2025-02-30", "31/04/2025", "2026-13-05", "00/00/0000"]):
    log(i, "order_date", "invalid_date", bad, "calendar-impossible date")
for i in pick("order_date", 3):
    log(i, "order_date", "future_date", f"2027-0{random.randint(1, 9)}-1{random.randint(0, 9)}",
        "order date after today")
for i in pick("signup_date", 6, lambda i: (i, "order_date") not in used):
    od = date.fromisoformat(orig[i]["order_date"])
    log(i, "signup_date", "logical_inconsistency",
        (od + timedelta(days=random.randint(5, 120))).isoformat(), "signup_date is after order_date")

# 3. numeric problems
for i, v in zip(pick("quantity", 6), ["-1", "-2", "0", "0", "250", "999"]):
    log(i, "quantity", "invalid_quantity", v, "negative, zero or absurd quantity")
for i, v in zip(pick("quantity", 4), ["two", "three", "3 pcs", "02"]):
    log(i, "quantity", "format_inconsistency", v, "quantity not a clean integer")
for i in pick("unit_price", 3, lambda i: True):
    log(i, "unit_price", "outlier", f"{float(orig[i]['unit_price']) * 100:.2f}", "x100 price outlier")
for i in pick("total_amount", 8):
    log(i, "total_amount", "calculation_mismatch",
        f"{float(orig[i]['total_amount']) * random.choice([0.9, 1.1, 1.5, 2.0]):.2f}",
        "total_amount != quantity * unit_price")
for i, v in zip(pick("rating", 8, lambda i: orig[i]["rating"] != ""),
                ["0", "6", "11", "-1", "five", "4.5", "7", "0"]):
    log(i, "rating", "invalid_value", v, "rating outside 1-5 or not an integer")


def money(v):
    n = int(round(v))
    return random.choice([f"\u20b9{v:,.2f}", f"Rs. {n}", f"{v:.2f} INR", f"{n:,}"])


for i in pick("unit_price", 40):
    log(i, "unit_price", "format_inconsistency", money(float(orig[i]["unit_price"])), "currency/format variant")
for i in pick("total_amount", 35):
    log(i, "total_amount", "format_inconsistency", money(float(orig[i]["total_amount"])), "currency/format variant")

# 4. emails
bad_email = [lambda e: e.replace("@", ""), lambda e: e.rsplit(".", 1)[0],
             lambda e: e.replace("@", "@@"), lambda e: e.replace(".com", ".con")]
for i in pick("email", 12):
    log(i, "email", "invalid_email", random.choice(bad_email)(orig[i]["email"]), "malformed email")
for i in pick("email", 15):
    log(i, "email", "format_inconsistency", orig[i]["email"].upper(), "uppercase email")

# 5. names
for i in pick("customer_name", 25):
    n = orig[i]["customer_name"]
    log(i, "customer_name", "format_inconsistency", random.choice([n.upper(), n.lower(), f"  {n} "]),
        "casing / stray whitespace")

# 6. phones
for i in pick("phone", 45):
    cu = cust_by_id[orig[i]["customer_id"]]
    code, d = cu["code"], cu["digits"]
    v = random.choice([f"{code}-{d}", f"{d[:5]} {d[5:]}", f"({code}) {d}", f"0{d}",
                       f"{d[:5]}-{d[5:]}", f"{code}{d}", d, d[:-1]])
    log(i, "phone", "invalid_phone" if v == d[:-1] else "format_inconsistency", v,
        "9 digits only" if v == d[:-1] else "phone format variant")

# 7. dates in mixed formats
DATEFMT = ["%d/%m/%Y", "%d %b %Y", "%B %d, %Y", "%Y/%m/%d", "%d-%m-%Y"]
for col, k in (("order_date", 160), ("signup_date", 90)):
    for i in pick(col, k):
        log(i, col, "format_inconsistency",
            date.fromisoformat(orig[i][col]).strftime(random.choice(DATEFMT)),
            "non-ISO date (DD/MM convention)")

# 8. categorical variants
CTRY = {"India": ["IN", "india", "INDIA", "Ind", "India ", " India"],
        "United Arab Emirates": ["UAE", "U.A.E", "United Arab Emirates ", "uae"],
        "Singapore": ["SG", "singapore", "SINGAPORE"],
        "United States": ["USA", "US", "united states", "U.S.A."]}
for i in pick("country", 40):
    log(i, "country", "format_inconsistency", random.choice(CTRY[orig[i]["country"]]), "country spelling variant")

ALIAS = {"Mumbai": "Bombay", "Bengaluru": "Bangalore", "Kolkata": "Calcutta",
         "Chennai": "Madras", "Gurugram": "Gurgaon", "Delhi": "New Delhi"}
for i in pick("city", 35):
    c = orig[i]["city"]
    if c in ALIAS and random.random() < 0.6:
        log(i, "city", "format_inconsistency", ALIAS[c], "alias / old name")
    else:
        log(i, "city", "format_inconsistency", random.choice([c.upper(), c.lower(), f"{c} "]), "casing / whitespace")

CATV = {"Electronics": ["electronics", "ELECTRONICS", "Electronic"], "Clothing": ["clothing", "Cloths", "CLOTHING"],
        "Home & Kitchen": ["Home and Kitchen", "Home&Kitchen", "home & kitchen"],
        "Books": ["books", "BOOKS", "Books "], "Beauty": ["beauty", "BEAUTY"],
        "Sports": ["sports", "Sports ", "SPORTS"], "Grocery": ["grocery", "GROCERY", "Groceries"]}
for i in pick("product_category", 35):
    log(i, "product_category", "format_inconsistency", random.choice(CATV[orig[i]["product_category"]]), "category variant")

PAYV = {"UPI": ["upi", "Upi", "UPI ", "U.P.I"], "Credit Card": ["credit card", "Credit_Card", "CC", "CREDIT CARD"],
        "Debit Card": ["debit card", "DC", "Debit_Card"], "Net Banking": ["Net banking", "NetBanking", "net-banking"],
        "Cash on Delivery": ["COD", "cod", "Cash On Delivery", "cash on delivery"]}
for i in pick("payment_method", 60):
    log(i, "payment_method", "format_inconsistency", random.choice(PAYV[orig[i]["payment_method"]]), "payment label variant")

STV = {"Delivered": ["delivered", "DELIVERED", "Deliverd"], "Shipped": ["shipped", "SHIPPED", " Shipped"],
       "Cancelled": ["canceled", "CANCELLED", "cancelled"], "Returned": ["returned", "RETURNED"],
       "Pending": ["pending", "PENDING", "Pendng"]}
for i in pick("status", 40):
    log(i, "status", "format_inconsistency", random.choice(STV[orig[i]["status"]]), "status variant / typo")

# ----------------------------------------------------------- duplicates
dup_rows, dup_issues = [], []
next_uid = N_ORDERS


def add_dup(row, typ, note, src):
    global next_uid
    row = dict(row)
    row["_uid"] = next_uid
    next_uid += 1
    dup_rows.append(row)
    dup_issues.append(dict(uid=row["_uid"], order_id=row["order_id"], column="*row*", issue_type=typ,
                           clean_value="", dirty_value="", note=f"{note} (source {base[src]['order_id']})"))
    return row


for i in random.sample(range(N_ORDERS), 25):
    add_dup(base[i], "exact_duplicate", "identical copy of another row", i)


def near_name(name):
    parts = name.strip().split()
    c = random.choice(["drop", "swap", "initial"])
    s = name.strip()
    if c == "drop":
        j = random.randint(1, len(s) - 2)
        return s[:j] + s[j + 1:]
    if c == "swap":
        j = random.randint(1, len(s) - 3)
        new = s[:j] + s[j + 1] + s[j] + s[j + 2:]
        if new != s:
            return new
    return f"{parts[0]} {random.choice('ABCDEFGHJKMNPRSTV')}. {parts[-1]}"


cands = [i for i in range(N_ORDERS) if len(base[i]["customer_name"].strip().split()) == 2
         and len(base[i]["customer_name"].strip()) > 8]
for i in random.sample(cands, 15):
    r = dict(base[i])
    r["customer_name"] = near_name(r["customer_name"])
    if "@" in r["email"] and random.random() < 0.5:
        r["email"] = r["email"].upper()
    add_dup(r, "near_duplicate", "same order, name typo/case variant", i)

for a, b in zip(random.sample(range(N_ORDERS), 5), random.sample(range(N_ORDERS), 5)):
    if a == b:
        continue
    r = dict(base[b])
    r["order_id"] = base[a]["order_id"]
    row = add_dup(r, "duplicate_key_conflict", "order_id reused with different content", a)
    dup_issues[-1].update(column="order_id", clean_value=base[b]["order_id"], dirty_value=base[a]["order_id"])

final = list(base)
for r in dup_rows:
    final.insert(random.randint(0, len(final)), r)

row_of = {r["_uid"]: n + 1 for n, r in enumerate(final)}   # 1-based data row (header excluded)

# ----------------------------------------------------------------- write
with open(f"{OUT}/orders_dirty.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(COLS)
    for r in final:
        w.writerow([r[c] for c in COLS])

with open(f"{OUT}/orders_ground_truth.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(COLS)
    for r in orig:
        w.writerow([r[c] for c in COLS])

all_issues = sorted(issues + dup_issues, key=lambda x: row_of[x["uid"]])
with open(f"{OUT}/planted_issues.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["data_row", "order_id", "column", "issue_type", "clean_value", "dirty_value", "note"])
    for x in all_issues:
        w.writerow([row_of[x["uid"]], x["order_id"], x["column"], x["issue_type"],
                    x["clean_value"], x["dirty_value"], x["note"]])

summary = dict(
    dirty_rows=len(final), clean_rows=N_ORDERS, columns=len(COLS),
    total_planted_issues=len(all_issues),
    by_issue_type=dict(Counter(x["issue_type"] for x in all_issues)),
    by_column=dict(Counter(x["column"] for x in all_issues)),
    legit_blank_ratings_not_errors=sum(1 for r in orig if r["rating"] == ""),
    note="Rating is legitimately blank for Pending/Cancelled/Shipped orders. Duplicate rows inherit the "
         "source row's other issues, which are not logged again. Dates use DD/MM convention.")
json.dump(summary, open(f"{OUT}/answer_key_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
