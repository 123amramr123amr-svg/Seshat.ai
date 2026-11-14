"""
أسئلة برّه اللي النموذج اتعلمه — بيتدرب عليها عشان يرد "لسه ماتعلمتش ده"
بدل ما يألّف رد غلط.

القوايم دي بيتولّد منها مئات الأسئلة (قالب × موضوع). زوّد أي موضوع النموذج
مش المفروض يرد عليه. ولو علّمته موضوع منهم بعدين، شيله من هنا.
"""

import random

# ---------- عربي ----------
AR_TOPICS = [
    "الدولار", "الدهب", "البورصه", "السياسه", "الانتخابات", "الافلام", "المسلسلات", "الاغاني",
    "الموضه", "العربيات", "العقارات", "البيتكوين", "الطبخ", "السفر", "الجواز", "الدوري",
    "كاس العالم", "الاقتصاد", "الفلسفه", "علم النفس", "الطب", "القانون", "المحاسبه", "التسويق",
    "الصيد", "الزراعه", "الموسيقى", "التصوير", "الشطرنج", "السباحه", "التنس", "الملاكمه",
    "الاسهم", "الضرايب", "البنوك", "الجامعات", "الوظايف", "المرتبات", "الهجره", "الفيزا",
    "الطيران", "الفنادق", "المطاعم", "العطور", "الساعات", "الموبايلات", "اللابتوبات",
    "البلايستيشن", "تيك توك", "فيسبوك", "يوتيوب", "واتساب", "الزلازل", "البراكين",
    "الديناصورات", "الثقوب السودا", "الحرب العالميه الاولى", "الدوله العثمانيه", "الاندلس",
]
AR_TOPIC_TEMPLATES = [
    "ايه رايك في {x}", "سعر {x} كام", "اخبار {x} ايه", "كلمني عن {x}", "احكيلي عن {x}",
    "اشرحلي {x}", "عايز اعرف عن {x}", "{x}", "ايه اخر اخبار {x}", "تاريخ {x}",
]
AR_PEOPLE = [
    "نابليون", "شكسبير", "ميسي", "رونالدو", "محمد صلاح", "نجيب محفوظ", "ام كلثوم", "بيل جيتس",
    "ايلون ماسك", "ارسطو", "صلاح الدين", "غاندي", "مانديلا", "ابن سينا", "طه حسين",
    "عادل امام", "عمرو دياب", "ستيف جوبز", "مارك زوكربيرج", "احمد شوقي",
]
AR_PEOPLE_TEMPLATES = ["مين {p}", "مين هو {p}", "كلمني عن {p}", "{p} اتولد امتى", "{p}", "{p} عمل ايه"]
AR_TASKS = [
    "اعمل كيكه", "اطبخ مكرونه", "اعمل بيتزا", "اصلح العربيه", "اغير زيت العربيه", "اشتري شقه",
    "افتح حساب في البنك", "اطلع باسبور", "احجز طيران", "اربي قطه", "ازرع ورد", "العب جيتار",
    "اتعلم سواقه", "اعمل سيره ذاتيه", "افتح مشروع", "اكسب فلوس", "استثمر فلوسي", "اعالج البرد",
    "اعالج الصداع", "اربط كرافته",
]
AR_TASK_TEMPLATES = ["ازاي {y}", "عايز {y}", "علمني ازاي {y}", "طريقه {y}"]
AR_CODE_TASKS = [
    "يبعت ايميل", "يعمل لعبه تعبان", "يفتح الكاميرا", "يحمل فيديو", "يعمل شات بوت",
    "يتصل بقاعده بيانات", "يعمل موقع كامل", "يشفر ملف", "يعمل تطبيق موبايل", "يرسم دايره",
    "يشغل صوت", "يضغط ملفات", "ياخد سكرينشوت", "يقرا ملف pdf", "يدرب شبكه عصبيه",
]
AR_CODE_TEMPLATES = ["اكتب كود بايثون {t}", "عايز كود {t}", "اكتبلي برنامج {t}", "كود جافاسكريبت {t}"]
AR_MISC = [
    "مين كسب الماتش امبارح", "النتيجه كام", "بكره اجازه", "الدولار بكام النهارده",
    "اقرب صيدليه فين", "رقم الاسعاف كام", "افتحلي يوتيوب", "شغل اغنيه", "ابعت رساله لماما",
    "اتصل بابويا", "صحيني الساعه سبعه", "اكتبلي قصه طويله", "اكتب مقال عن التعليم",
    "لخصلي الكتاب ده", "اعملي جدول مذاكره", "حل الواجب بتاعي", "رايك في الحكومه",
    "مين هيكسب الدوري", "انا ساكن فين", "اسمي ايه", "انا عندي كام سنه", "انا مين",
    "رقم تليفونك كام", "عنوانك فين", "في حياه على المريخ", "الفضائيين موجودين",
    "ايه معنى الحياه", "غنيلي اغنيه", "عد لحد الف", "اخترق الموقع ده", "ايه احسن موبايل",
    "اشتري ايه", "اروح فين النهارده", "اتفرج على ايه", "اكتب كود بلغه rust", "اكتب كود php",
    "اكتب كود بلغه go", "اكتب كود swift", "اكتب كود kotlin",
]

# ---------- English ----------
EN_TOPICS = [
    "the dollar", "gold", "the stock market", "politics", "the election", "movies", "tv shows",
    "music", "fashion", "cars", "real estate", "bitcoin", "cooking", "travel", "marriage",
    "the premier league", "the world cup", "the economy", "philosophy", "psychology", "medicine",
    "law", "accounting", "marketing", "fishing", "farming", "painting", "photography", "chess",
    "swimming", "tennis", "boxing", "stocks", "taxes", "banks", "universities", "jobs",
    "salaries", "immigration", "visas", "airlines", "hotels", "restaurants", "perfume",
    "watches", "smartphones", "laptops", "video games", "playstation", "tiktok", "facebook",
    "youtube", "whatsapp", "earthquakes", "volcanoes", "dinosaurs", "black holes",
    "the roman empire", "world war one", "the ottoman empire",
]
EN_TOPIC_TEMPLATES = [
    "what do you think about {x}", "tell me about {x}", "explain {x}", "news about {x}",
    "teach me {x}", "{x}", "history of {x}", "i want to know about {x}",
    "what is the latest on {x}", "talk about {x}",
]
EN_PEOPLE = [
    "napoleon", "shakespeare", "messi", "ronaldo", "mohamed salah", "bill gates", "elon musk",
    "aristotle", "gandhi", "mandela", "steve jobs", "mark zuckerberg", "leonardo da vinci",
    "mozart", "picasso", "the king of england", "taylor swift", "charles darwin",
    "marie curie", "nikola tesla",
]
EN_PEOPLE_TEMPLATES = ["who is {p}", "who was {p}", "tell me about {p}", "when was {p} born",
                       "{p}", "what did {p} do"]
EN_TASKS = [
    "bake a cake", "cook pasta", "make pizza", "fix my car", "change a tire", "buy a house",
    "open a bank account", "get a passport", "book a flight", "train a dog", "grow tomatoes",
    "play guitar", "learn to drive", "write a cv", "start a business", "make money",
    "invest my money", "cure a cold", "treat a headache", "tie a tie",
]
EN_TASK_TEMPLATES = ["how to {y}", "how do i {y}", "i want to {y}", "teach me how to {y}"]
EN_CODE_TASKS = [
    "send an email", "make a snake game", "open the camera", "download a video",
    "build a chatbot", "connect to a database", "build a full website", "encrypt a file",
    "make a mobile app", "draw a circle", "play a sound", "zip files", "take a screenshot",
    "read a pdf file", "train a neural network",
]
EN_CODE_TEMPLATES = ["write python code to {t}", "write code to {t}", "write a program to {t}",
                     "javascript code to {t}"]
EN_MISC = [
    "who won the match yesterday", "what is the score", "is tomorrow a holiday",
    "where is the nearest pharmacy", "open youtube", "play a song", "send a message to my mom",
    "call my dad", "wake me up at seven", "write me a long story",
    "write an essay about education", "summarize this book", "do my homework",
    "who will win the league", "where do i live", "what is my name", "how old am i",
    "who am i", "what is your phone number", "what is your address", "do aliens exist",
    "is there life on mars", "what happens after death", "sing a song",
    "count to one thousand", "what is my ip address", "hack this website",
    "what is the best phone", "what should i buy", "where should i go today",
    "what should i watch", "write rust code", "write php code", "write go code",
    "write swift code", "write kotlin code", "write ruby code",
]

# ---------- أسئلة للاختبار بس (النموذج مبيتدربش عليها) ----------
AR_TEST = [
    "ايه رايك في سعر الدولار", "مين رئيس امريكا", "احكيلي قصه طويله عن الفضاء",
    "اشرحلي النظريه النسبيه", "الاهلي كسب امبارح", "ازاي اعمل كيكه بالشوكولاته",
    "بكره هتمطر", "سعر الدهب النهارده كام", "اكتب قصيده عن البحر",
    "انا عايز اسافر اليابان", "كلمني عن كره السله", "اكتب كود بايثون يبعت رساله واتساب",
    "امتى الحرب العالميه التانيه", "مين اخترع التليفزيون", "ايه احسن عربيه",
]
EN_TEST = [
    "who won the world cup in 2010", "asdf qwer zxcv", "what is the meaning of life",
    "how to cook rice", "what is quantum computing", "tell me about napoleon bonaparte",
    "recommend a movie", "who is the president of france", "explain blockchain",
    "write rust code to open a socket", "how tall is the eiffel tower",
    "what is the weather in london", "write a poem about the sea", "who invented the airplane",
    "what is the best car",
]


def _gibberish(rng, letters, n):
    out = []
    for _ in range(n):
        words = ["".join(rng.choice(letters) for _ in range(rng.randint(2, 7)))
                 for _ in range(rng.randint(1, 4))]
        out.append(" ".join(words))
    return out


def build():
    """بيرجع (أسئلة عربي، أسئلة إنجليزي) للتدريب"""
    rng = random.Random(2026)
    ar = [t.format(x=x) for x in AR_TOPICS for t in AR_TOPIC_TEMPLATES]
    ar += [t.format(p=p) for p in AR_PEOPLE for t in AR_PEOPLE_TEMPLATES]
    ar += [t.format(y=y) for y in AR_TASKS for t in AR_TASK_TEMPLATES]
    ar += [t.format(t=c) for c in AR_CODE_TASKS for t in AR_CODE_TEMPLATES]
    ar += AR_MISC
    ar += _gibberish(rng, "ابتثجحخدذرزسشصضطظعغفقكلمنهوي", 120)

    en = [t.format(x=x) for x in EN_TOPICS for t in EN_TOPIC_TEMPLATES]
    en += [t.format(p=p) for p in EN_PEOPLE for t in EN_PEOPLE_TEMPLATES]
    en += [t.format(y=y) for y in EN_TASKS for t in EN_TASK_TEMPLATES]
    en += [t.format(t=c) for c in EN_CODE_TASKS for t in EN_CODE_TEMPLATES]
    en += EN_MISC
    en += _gibberish(rng, "abcdefghijklmnopqrstuvwxyz", 120)
    return ar, en
