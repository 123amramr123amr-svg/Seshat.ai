"""
فهم الأسئلة اللي برّه معلومات seshat — وتكوين رد عليها
========================================================

قبل كده أي سؤال برّه اللي اتعلمه كان بياخد نفس الجملة ("لسه ماتعلمتش الرد…").
دلوقتي النموذج بيتدرب يفهم حاجتين في السؤال:

    1. نوعه:    رأي؟ سعر؟ أخبار؟ سؤال عن شخص؟ طلب تنفيذ؟ توقع؟ …
    2. موضوعه:  الكلمات اللي السؤال بيدور حواليها — بينسخها من السؤال حرف حرف.

وبعدين يكوّن رد فيه الموضوع نفسه. سطر التفكير بيبقى:   النوع | الموضوع
مثال (الموضوع "القهوه" عمره ما ظهر في التدريب):

    السؤال:   ايه رايك في القهوه
    التفكير:  رأي | القهوه
    الرد:     بتسألني عن رأيي في القهوه. أنا نموذج ذكاء اصطناعي ومعنديش رأي شخصي …

يعني الجملة دي متكونة من فهم السؤال، مش محفوظة: النموذج عمره ما شافها.

الملف ده فيه: أنواع الأسئلة وقوالبها، مواضيع التدريب، مواضيع الاختبار (HELD_OUT_*)
اللي النموذج مبيشوفهاش، و compose / detect اللي المحرك بيراجع بيهم شغل النموذج.
"""

import difflib
import random
import re

# ============================================================
# أنواع فيها موضوع: (المفتاح، اسم عربي، اسم إنجليزي، قوالب عربي، قوالب إنجليزي، رد عربي، رد إنجليزي)
# {x} = الموضوع. أول قالب هو الصيغة الأساسية.
# ============================================================
TOPIC_KINDS = [
    ("opinion", "رأي", "opinion",
     ["ايه رايك في {x}", "رايك ايه في {x}", "رايك في {x}", "شايف {x} ازاي", "بتحب {x}",
      "ايه رايك ف {x}", "قولي رايك في {x}"],
     ["what do you think about {x}", "what is your opinion on {x}", "do you like {x}",
      "your thoughts on {x}", "what do you think of {x}"],
     "بتسألني عن رأيي في {x}. أنا نموذج ذكاء اصطناعي ومعنديش رأي شخصي ولا ذوق، فمقدرش أحكم على {x}.",
     "You are asking what I think about {x}. I am an AI model with no personal opinions or taste, so I cannot judge {x}."),
    ("price", "سعر", "price",
     ["سعر {x} كام", "{x} بكام", "بكام {x}", "سعر {x} النهارده", "{x} سعره كام", "ثمن {x} كام"],
     ["what is the price of {x}", "how much is {x}", "how much does {x} cost", "price of {x}",
      "{x} price today"],
     "بتسأل عن سعر {x}. الأسعار بتتغير كل يوم، فمقدرش أقولك سعر {x} من غير مصدر جديد.",
     "You are asking about the price of {x}. Prices change every day, so I cannot give the price of {x} without a fresh source."),
    ("news", "أخبار", "news",
     ["اخبار {x} ايه", "ايه اخر اخبار {x}", "اخر اخبار {x}", "ايه الجديد في {x}", "ايه اخبار {x}"],
     ["what is the latest news about {x}", "news about {x}", "what is the latest on {x}",
      "what is new in {x}", "any news about {x}"],
     "بتسأل عن أخبار {x}. الأخبار بتتغير كل ساعة، ومعلوماتي مش لحظية عشان أقولك الجديد في {x}.",
     "You are asking for news about {x}. News changes every hour, and my knowledge is not live, so I cannot tell you what is new in {x}."),
    ("about", "موضوع", "topic",
     ["كلمني عن {x}", "احكيلي عن {x}", "اشرحلي {x}", "عايز اعرف عن {x}", "معلومات عن {x}",
      "تاريخ {x}", "ايه هو {x}", "ايه هي {x}", "تعرف ايه عن {x}", "{x}"],
     ["tell me about {x}", "explain {x}", "i want to know about {x}", "history of {x}",
      "talk about {x}", "teach me {x}", "what do you know about {x}", "{x}"],
     "بتسأل عن {x}. معلوماتي عن {x} لسه مش كفاية عشان أشرح صح، ومش هألف كلام من عندي.",
     "You are asking about {x}. I do not know enough about {x} to explain it correctly, and I will not make things up."),
    ("person", "شخص", "person",
     ["مين {x}", "مين هو {x}", "{x} اتولد امتى", "{x} عمل ايه", "{x} مين", "تعرف {x}"],
     ["who is {x}", "who was {x}", "when was {x} born", "what did {x} do", "do you know {x}"],
     "بتسأل عن {x}. معنديش معلومات مؤكدة عن {x}، ومش هألف كلام من عندي.",
     "You are asking about {x}. I have no reliable information about {x}, and I will not make things up."),
    ("howto", "طريقة", "how-to",
     ["ازاي {x}", "علمني ازاي {x}", "طريقه {x}", "عايز اعرف ازاي {x}", "ايه طريقه {x}"],
     ["how to {x}", "how do i {x}", "teach me how to {x}", "i want to {x}", "how can i {x}"],
     "عايز تعرف طريقة: {x}. دي حاجة محتاجة خطوات من مصدر موثوق، ومعلوماتي عنها لسه مش كفاية.",
     "You want to know how to {x}. That needs steps from a reliable source, and I do not know enough about it yet."),
    ("code", "كود كبير", "big code",
     ["اكتب كود بايثون {x}", "عايز كود {x}", "اكتبلي برنامج {x}", "كود جافاسكريبت {x}",
      "اكتب كود {x}"],
     ["write python code to {x}", "write code to {x}", "write a program to {x}",
      "javascript code to {x}", "code to {x}"],
     "عايز كود: {x}. ده أكبر من الأكواد اللي بعرف أكتبها دلوقتي؛ بكتب أكواد قصيرة زي المضروب وعكس النص وترتيب القايمة.",
     "You want code to {x}. That is bigger than what I can write now; I write short programs like factorial, reversing a string or sorting a list."),
    ("best", "اختيار", "choice",
     ["ايه احسن {x}", "احسن {x} ايه", "رشحلي {x}", "انهي {x} احسن", "اختارلي {x}"],
     ["what is the best {x}", "recommend a {x}", "which {x} is the best", "suggest a good {x}"],
     "بتطلب مني أختارلك أحسن {x}. الاختيار بيعتمد على ذوقك وميزانيتك واحتياجك، فمقدرش أحدد أحسن {x} ليك.",
     "You are asking me to pick the best {x}. That depends on your taste, budget and needs, so I cannot choose the best {x} for you."),
    ("write", "كتابة طويلة", "long writing",
     ["اكتب مقال عن {x}", "اكتب قصه عن {x}", "اكتب قصيده عن {x}", "اكتبلي موضوع تعبير عن {x}",
      "الف قصه عن {x}"],
     ["write an essay about {x}", "write a story about {x}", "write a poem about {x}",
      "write an article about {x}"],
     "بتطلب مني كتابة طويلة عن {x}. أنا لسه بكتب ردود قصيرة وأكواد صغيرة، ومقدرش أكتب نص طويل عن {x}.",
     "You are asking for a long piece of writing about {x}. I only write short replies and small programs so far, so I cannot write a long text about {x}."),
]

# ============================================================
# أنواع من غير موضوع: (المفتاح، اسم عربي، اسم إنجليزي، وصف عربي، وصف إنجليزي، أسئلة عربي، أسئلة إنجليزي، رد عربي، رد إنجليزي)
# الوصف هو اللي بيتكتب بعد | في سطر التفكير.
# ============================================================
FIXED_KINDS = [
    ("action", "تنفيذ", "action", "طلب تنفيذ على الجهاز", "a request to act on the device",
     ["افتحلي يوتيوب", "شغل اغنيه", "ابعت رساله لماما", "اتصل بابويا", "صحيني الساعه سبعه",
      "افتح الكاميرا", "اقفل الموبايل", "حملي الفيديو ده", "افتح واتساب", "شغل الموسيقى",
      "ابعت ايميل لمديري", "اظبط المنبه", "افتح جوجل", "شغل الفيلم", "اتصل بالدكتور",
      "افتح الفيسبوك", "علي الصوت", "وطي الصوت", "اطفي النور", "شغل النور", "ابعتلي الملف",
      "صورني", "سجل صوتي", "افتح الخريطه", "اطلبلي اكل", "احجزلي تذكره", "ذكرني بكره"],
     ["open youtube", "play a song", "send a message to my mom", "call my dad",
      "wake me up at seven", "open the camera", "turn off the phone", "download this video",
      "open whatsapp", "play music", "send an email to my boss", "set an alarm", "open google",
      "play the movie", "call the doctor", "open facebook", "turn up the volume",
      "turn off the lights", "turn on the lights", "take a photo", "record my voice",
      "open the map", "order food for me", "book a ticket for me", "remind me tomorrow"],
     "ده طلب تنفيذ على جهازك. أنا شات بيرد بالكلام بس، ومقدرش أفتح تطبيقات ولا أتصل ولا أبعت رسايل.",
     "That is a request to do something on your device. I am a chat that only replies with text; I cannot open apps, make calls or send messages."),
    ("you", "عن المستخدم", "about the user", "سؤال عن المستخدم نفسه", "a question about the user",
     ["انا ساكن فين", "اسمي ايه", "انا عندي كام سنه", "انا مين", "عيد ميلادي امتى",
      "انا بشتغل ايه", "انا شكلي ايه", "انا فين دلوقتي", "رقم تليفوني كام", "انا متجوز",
      "عارف انا مين", "تعرف اسمي", "فاكرني", "انا قلتلك ايه امبارح", "احنا اتكلمنا قبل كده",
      "فاكر كلامنا", "انا من انهي بلد", "ايه هوايتي", "انا بحب ايه"],
     ["where do i live", "what is my name", "how old am i", "who am i", "when is my birthday",
      "what is my job", "what do i look like", "where am i now", "what is my phone number",
      "am i married", "do you know who i am", "do you know my name", "do you remember me",
      "what did i tell you yesterday", "have we talked before", "which country am i from",
      "what is my hobby", "what do i like"],
     "بتسألني عن حاجة تخصك انت. أنا مبعرفش عنك غير اللي بتكتبه في الرسالة، ومبفتكرش الرسايل اللي فاتت.",
     "You are asking about yourself. I know nothing about you except what you write in the message, and I do not remember earlier messages."),
    ("future", "توقع", "prediction", "سؤال عن حاجة لسه محصلتش", "a question about the future",
     ["مين هيكسب الدوري", "بكره هتمطر", "الدولار هيوصل كام", "مين هيكسب الماتش",
      "هنجح في الامتحان", "هتجوز امتى", "مين هيكسب كاس العالم", "الاسعار هتغلى",
      "هيحصل ايه بكره", "ايه اللي هيحصل السنه الجايه", "هبقى غني", "الدهب هيغلى",
      "مين هيفوز في الانتخابات", "هموت امتى", "الجو هيبقى عامل ايه بكره", "هلاقي شغل",
      "مستقبلي ايه", "العالم هيخلص امتى"],
     ["who will win the league", "will it rain tomorrow", "how high will the dollar go",
      "who will win the match", "will i pass the exam", "when will i get married",
      "who will win the world cup", "will prices go up", "what will happen tomorrow",
      "what will happen next year", "will i be rich", "will gold go up",
      "who will win the election", "when will i die", "will i find a job",
      "what is my future", "when will the world end"],
     "ده سؤال عن حاجة لسه محصلتش. مفيش حد يقدر يعرف المستقبل، وأنا مش هخمّن وأقولك كلام مش مؤكد.",
     "That is a question about something that has not happened yet. Nobody can know the future, and I will not guess."),
    ("live", "معلومة لحظية", "live info", "سؤال محتاج معلومة من دلوقتي أو من مكانك",
     "a question that needs live or local information",
     ["مين كسب الماتش امبارح", "النتيجه كام", "بكره اجازه", "اقرب صيدليه فين",
      "رقم الاسعاف كام", "الماتش الساعه كام", "الاهلي كسب", "الطريق زحمه",
      "اقرب مطعم فين", "المحل فاتح", "القطر هييجي امتى", "في مدرسه بكره", "الامتحان امتى",
      "اقرب بنك فين", "النت واقع ليه", "الكهربا هتيجي امتى", "ايه اللي بيتعرض دلوقتي",
      "اقرب مستشفى فين", "نتيجه الثانويه طلعت"],
     ["who won the match yesterday", "what is the score", "is tomorrow a holiday",
      "where is the nearest pharmacy", "what time is the match", "is the road busy",
      "where is the nearest restaurant", "is the shop open", "when does the train arrive",
      "is there school tomorrow", "when is the exam", "where is the nearest bank",
      "why is the internet down", "what is on tv now", "where is the nearest hospital",
      "what is my ip address"],
     "ده سؤال محتاج معلومة من دلوقتي أو من مكانك (نتيجة، ميعاد، مكان قريب). أنا مبشوفش مكانك ومعلوماتي مش لحظية.",
     "That question needs live or local information (a score, a time, a nearby place). I cannot see where you are and my knowledge is not live."),
    ("pick", "قرار شخصي", "personal choice", "طلب اختيار حاجة شخصية", "a request to choose for the user",
     ["اشتري ايه", "اروح فين النهارده", "اتفرج على ايه", "اكل ايه النهارده", "البس ايه",
      "اعمل ايه دلوقتي", "اسافر فين", "اذاكر ايه", "اقرا ايه", "العب ايه", "اسمي ابني ايه",
      "اجيب هديه ايه", "اخرج ولا اقعد", "اختار ايه", "اشتغل ايه", "ادخل كليه ايه"],
     ["what should i buy", "where should i go today", "what should i watch",
      "what should i eat today", "what should i wear", "what should i do now",
      "where should i travel", "what should i study", "what should i read",
      "what should i play", "what should i name my son", "what gift should i get",
      "should i go out or stay home", "what should i choose", "which job should i take"],
     "بتطلب مني أقرر حاجة شخصية ليك. ده بيرجع لذوقك وظروفك انت، وأنا مقدرش أقرر مكانك.",
     "You are asking me to make a personal choice for you. That depends on your own taste and situation, so I cannot decide for you."),
    ("long", "كتابة طويلة", "long writing", "طلب كتابة طويلة", "a request for long writing",
     ["اكتبلي قصه طويله", "لخصلي الكتاب ده", "اعملي جدول مذاكره", "حل الواجب بتاعي",
      "اكتبلي بحث", "اكتب خطاب", "ترجملي الصفحه دي", "اكتبلي سيره ذاتيه", "اكتب روايه",
      "لخصلي الفيلم ده", "اكتبلي خطه", "اكتب كلمه للحفله", "اكتبلي ايميل طويل", "حللي المساله دي",
      "اكتبلي تقرير", "عد لحد الف", "غنيلي اغنيه"],
     ["write me a long story", "summarize this book", "make me a study schedule",
      "do my homework", "write me a research paper", "write a letter", "translate this page",
      "write my cv", "write a novel", "summarize this movie", "write me a plan",
      "write a speech", "write me a long email", "write me a report", "count to one thousand",
      "sing a song"],
     "ده طلب كتابة أو شغل طويل. أنا لسه بكتب ردود قصيرة وأكواد صغيرة، ومقدرش أعمل نص طويل دلوقتي.",
     "That is a request for long writing or a long task. I only write short replies and small programs so far, so I cannot produce a long text yet."),
    ("open", "سؤال مفتوح", "open question", "سؤال ملوش إجابة واحدة مؤكدة", "a question with no single sure answer",
     ["ايه معنى الحياه", "في حياه على المريخ", "الفضائيين موجودين", "ايه اللي بيحصل بعد الموت",
      "احنا عايشين ليه", "ايه سر السعاده", "الكون ليه نهايه", "ايه هو الحب الحقيقي",
      "مين احسن لاعب في التاريخ", "الفلوس بتجيب سعاده", "ايه اصل الكون", "الاحلام معناها ايه",
      "في عوالم تانيه", "ايه الصح وايه الغلط"],
     ["what is the meaning of life", "is there life on mars", "do aliens exist",
      "what happens after death", "why are we here", "what is the secret of happiness",
      "does the universe have an end", "what is true love",
      "who is the best player in history", "does money buy happiness",
      "what do dreams mean", "are there other worlds", "what is right and wrong"],
     "ده سؤال كبير ملوش إجابة واحدة مؤكدة، والناس مختلفين فيه. أنا مش هدّعي إني عارف الإجابة النهائية.",
     "That is a big question with no single sure answer, and people disagree about it. I will not pretend to know the final answer."),
    ("harm", "طلب مؤذي", "harmful request", "طلب ممكن يأذي حد", "a request that could hurt someone",
     ["اخترق الموقع ده", "اخترق حساب صاحبي", "اسرق باسورد", "اعمل فيروس", "هكرلي الواي فاي",
      "ازاي اغش في الامتحان", "اخترق الفيسبوك", "اتجسس على موبايل", "ازاي اسرق",
      "اعمل ايميل مزيف", "هكر اللعبه"],
     ["hack this website", "hack my friend account", "steal a password", "make a virus",
      "hack the wifi", "how to cheat in the exam", "hack facebook", "spy on a phone",
      "how to steal", "make a fake email", "hack the game"],
     "ده طلب ممكن يأذي حد أو يخالف القانون، فمش هساعد فيه. لو عايز تتعلم الحماية والأمان أنا معاك.",
     "That request could hurt someone or break the law, so I will not help with it. If you want to learn about security and protection, I am with you."),
    ("unclear", "مش مفهوم", "unclear", "كلام مش واضح", "text that is not clear",
     [], [],
     "مفهمتش الكلام ده. ممكن تكتبه بكلمات تانية؟",
     "I could not understand that. Could you write it in other words?"),
]

KINDS = {k[0]: k for k in TOPIC_KINDS}
FIXED = {k[0]: k for k in FIXED_KINDS}

# الأنواع اللي البحث في النت ممكن يجاوبها (معلومة عن موضوع أو شخص أو طريقة).
# الباقي ردّه معروف من نوعه (رأي، توقع، طلب تنفيذ…) والبحث فيه بيجيب كلام ملوش علاقة.
SEARCHABLE = {"about", "person"}

# ============================================================
# مواضيع التدريب
# ============================================================
AR_EXTRA_TOPICS = [
    "الفراعنه الجدد", "السينما", "المسرح", "الروايات", "الشعر", "الرسم", "النحت", "الخط العربي",
    "الفلك", "الجيولوجيا", "المحيطات", "الغابات", "الصحرا", "القطب الشمالي", "الحيتان",
    "النحل", "النمل", "الصقور", "الخيل", "الجمال", "الكلاب", "الببغاوات", "الاسماك",
    "العسل", "الخبز", "الجبنه", "الفول", "الكشري", "الملوخيه", "العصير", "الشوكولاته",
    "الدراجات", "القطارات", "السفن", "الصواريخ", "الطيارات الورق", "الروبوتات الصغيره",
    "الهندسه", "الكيميا", "الاحياء", "الجبر", "الاحصاء", "المنطق", "علم الاجتماع",
    "الصحافه", "الاذاعه", "الاعلانات", "التجاره", "الصناعه", "السياحه", "الاثار الرومانيه",
    "الخزف", "النجاره", "الحداده", "الخياطه", "التطريز", "البستنه", "تربيه الدواجن",
    "كره اليد", "الجمباز", "الجري", "المصارعه", "رفع الاثقال", "اليوجا", "التاملات",
    "الذكاء العاطفي", "ادارة الوقت", "القياده", "ريادة الاعمال", "العمل الحر", "التدوين",
]
EN_EXTRA_TOPICS = [
    "cinema", "theatre", "novels", "poetry", "painting", "sculpture", "calligraphy",
    "astronomy", "geology", "the oceans", "forests", "deserts", "the north pole", "whales",
    "bees", "ants", "falcons", "horses", "camels", "dogs", "parrots", "fish",
    "honey", "bread", "cheese", "pasta", "pizza", "juice", "chocolate", "ice cream",
    "bicycles", "trains", "ships", "rockets", "kites", "drones",
    "engineering", "chemistry", "biology", "algebra", "statistics", "logic", "sociology",
    "journalism", "radio", "advertising", "trade", "industry", "tourism", "roman ruins",
    "pottery", "carpentry", "sewing", "gardening", "farming",
    "handball", "gymnastics", "running", "wrestling", "weightlifting", "yoga", "meditation",
    "time management", "leadership", "startups", "freelancing", "blogging", "podcasts",
]
AR_FIRST = ["احمد", "محمود", "علي", "حسن", "يوسف", "خالد", "عمر", "ابراهيم", "مصطفى", "كريم",
            "ساره", "مريم", "نور", "هدى", "ليلى", "منى", "فاطمه", "زينب", "سلمى", "دينا"]
AR_LAST = ["المصري", "الشريف", "عبد الله", "حسين", "سليمان", "النجار", "فهمي", "رشدي",
           "الصاوي", "عثمان", "البنا", "شاهين", "زكي", "منصور", "الجندي", "سالم"]
EN_FIRST = ["john", "mary", "david", "anna", "peter", "linda", "james", "sophie", "daniel",
            "emma", "thomas", "laura", "robert", "julia", "michael", "sarah", "george", "alice"]
EN_LAST = ["smith", "brown", "taylor", "wilson", "clark", "lewis", "walker", "hall", "young",
           "king", "wright", "green", "baker", "adams", "nelson", "carter"]
AR_EXTRA_TASKS = [
    "اعمل قهوه", "اغسل العربيه", "ارتب الاوضه", "احفظ بسرعه", "انام بدري", "اخس عشره كيلو",
    "اتعلم انجليزي", "اعمل عصير", "اصلح الحنفيه", "اركب دراجه", "اكتب بخط حلو", "اذاكر كويس",
    "اعمل قناه يوتيوب", "اصور فيديو", "اربي عصافير", "اعوم", "اطير طياره ورق", "اعمل صابون",
]
EN_EXTRA_TASKS = [
    "make coffee", "wash the car", "clean my room", "memorize fast", "sleep early",
    "lose ten kilos", "learn spanish", "make juice", "fix a tap", "ride a bike",
    "improve my handwriting", "study well", "start a youtube channel", "shoot a video",
    "keep birds", "swim", "fly a kite", "make soap",
]
AR_THINGS = ["موبايل", "لابتوب", "عربيه", "ساعه", "سماعه", "تابلت", "شاشه", "كاميرا", "كتاب",
             "فيلم", "مسلسل", "لعبه", "مطعم", "فندق", "جامعه", "مدرسه", "عطر", "شنطه", "جزمه",
             "قهوه", "شاي", "كليه", "شغلانه", "هديه", "برنامج", "متصفح", "لغه برمجه"]
EN_THINGS = ["phone", "laptop", "car", "watch", "headset", "tablet", "screen", "camera", "book",
             "movie", "tv show", "game", "restaurant", "hotel", "university", "school",
             "perfume", "bag", "pair of shoes", "coffee", "tea", "job", "gift", "app", "browser",
             "programming language"]

# ---------- للاختبار بس: مواضيع النموذج عمره ما شافها ----------
HELD_OUT_AR = ["القهوه", "الشاي", "المترو", "الكوره", "كره السله", "التعليم", "النظريه النسبيه",
               "الطاقه الشمسيه", "السوشيال ميديا", "العملات الرقميه", "القطط", "الواحات"]
HELD_OUT_EN = ["coffee beans", "the subway", "basketball", "quantum computing", "blockchain",
               "solar energy", "social media", "street food", "volcano islands", "board games"]
HELD_OUT_AR_PEOPLE = ["احمد زويل", "مجدي يعقوب", "فيروز", "جمال عبد الناصر", "هاني شاكر",
                      "سميره موسى"]
HELD_OUT_EN_PEOPLE = ["marie curie", "nikola tesla", "charles darwin", "frida kahlo",
                      "alan turing", "jane austen"]
HELD_OUT_AR_TASKS = ["اعمل كيكه بالشوكولاته", "اسافر اليابان", "اطبخ رز", "اصلح الموبايل"]
HELD_OUT_EN_TASKS = ["cook rice", "open a socket", "fix my phone", "bake a chocolate cake"]


def _letters_words(rng, letters, n):
    """كلمات من حروف عشوائية — عشان النموذج يتعلم ينسخ الموضوع حرف حرف مش يحفظ مواضيع"""
    out = []
    for _ in range(n):
        words = ["".join(rng.choice(letters) for _ in range(rng.randint(3, 7)))
                 for _ in range(rng.choice((1, 1, 2)))]
        out.append(" ".join(words))
    return out


AR_LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
EN_LETTERS = "abcdefghijklmnopqrstuvwxyz"


_FILLER = set("""يا في من على عن ال هو هي ده دي ايه اي ما هل كم كام لو سمحت ممكن طيب بقى بقا
كده قولي قوللي عايز عاوز اعرف هى انت هو ليه امتى فين مين ازاي
the a an of is are was were what how many much please tell me do does did you know in on
to for why when where who which""".split())

# كلمات السؤال: وجودها معناه إن ده سؤال عن معلومة حتى لو مفيش قالب اتطابق
_ASK = set("""كم كام ليه لماذا امتى متى فين اين مين ازاي كيف ايه ما هل
what why when where who which how is are was were does do did can""".split())


def content_words(text):
    """الكلمات اللي شايلة المعنى (من غير أدوات السؤال وحروف الجر)"""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return {w for w in text.split() if len(w) >= 3 and w not in _FILLER}


def _bare(word):
    return word[2:] if word.startswith("ال") and len(word) > 4 else word


def other_subject(question, known):
    """
    السؤال والسؤال المعروف نفس الجملة بس كلمة واحدة مهمة اتبدلت بكلمة تانية خالص؟
    ("مين اخترع التليفون" × "مين اخترع الصفر"، "كم عدد سكان الصين" × "عدد سكان مصر").
    ساعتها رد السؤال المعروف يبقى إجابة عن حاجة تانية، فمينفعش يتقال.
    الصيغ المختلفة لنفس السؤال ("مين الشخص اللي برمجك" × "مين طورك") مش بتتحسب: دي مفيهاش
    كلام مشترك والفرق أكتر من كلمة. والكلمتين القريبين من بعض ("بتغلي" × "غليان") نفس المعنى.
    """
    q, k = content_words(question), content_words(known)
    if not (q & k) or len(q - k) != 1 or len(k - q) != 1:
        return False
    a, b = _bare(next(iter(q - k))), _bare(next(iter(k - q)))
    return difflib.SequenceMatcher(None, a, b).ratio() < 0.5


def is_fact_question(question):
    """سؤال فيه أداة استفهام وكلمة معنى واحدة على الأقل → سؤال عن معلومة"""
    return bool(set(question.split()) & _ASK) and bool(content_words(question))


def label(kind, arabic):
    if kind == "fact":
        return "سؤال عن معلومة" if arabic else "fact question"
    k = KINDS.get(kind) or FIXED[kind]
    return k[1] if arabic else k[2]


def think(kind, topic, arabic):
    """سطر التفكير: النوع | الموضوع (أو وصف النوع لو ملوش موضوع)"""
    if kind in KINDS:
        return f"{label(kind, arabic)} | {topic}"
    k = FIXED[kind]
    return f"{label(kind, arabic)} | {k[3] if arabic else k[4]}"


def compose(kind, topic, arabic):
    """الرد اللي بيتكوّن من النوع والموضوع"""
    if kind in KINDS:
        k = KINDS[kind]
        return (k[5] if arabic else k[6]).replace("{x}", topic)
    k = FIXED[kind]
    return k[7] if arabic else k[8]


def parse_think(thinking):
    """سطر تفكير → (النوع، الموضوع، عربي؟) أو None لو مش من أنواع الفهم"""
    if " | " not in thinking:
        return None
    name, rest = thinking.split(" | ", 1)
    name, rest = name.strip(), rest.strip()
    for key, k in KINDS.items():
        if name in (k[1], k[2]) and rest:
            return key, rest, name == k[1]
    for key, k in FIXED.items():
        if (name, rest) in ((k[1], k[3]), (k[2], k[4])):
            return key, "", name == k[1]
    return None


def _template_regex(template):
    head, tail = template.split("{x}")
    return re.compile("^" + re.escape(head) + r"(.{2,80})" + re.escape(tail) + "$")


def _norm_template(t):
    """القالب بنفس تنضيف الأسئلة (ى → ي، ة → ه…) عشان يطابق السؤال بعد normalize_question"""
    try:
        from tokenizer import normalize_question
    except Exception:
        return t
    return normalize_question(t.replace("{x}", "QXQ")).replace("qxq", "{x}").replace("QXQ", "{x}")


# قوالب الكشف بالقواعد: الأطول الأول. القالب "{x}" لوحده مبيدخلش (أي جملة هتطابقه).
_DETECT = sorted(
    ((kind, arabic, t, _template_regex(_norm_template(t)))
     for kind, _, _, ar, en, _, _ in TOPIC_KINDS
     for arabic, templates in ((True, ar), (False, en))
     for t in templates if t != "{x}"),
    key=lambda item: -len(item[2]))


def detect(question):
    """
    كشف بالقواعد (من غير النموذج): بيطابق السؤال المتنضف مع القوالب.
    بيرجع (النوع، الموضوع) أو None. المحرك بيستخدمه لما النموذج ميطلعش فهم سليم.
    """
    plain = _plain(question)                   # من غير "ممكن / لو سمحت / اكتبلي → اكتب"
    found = _detect(plain) if plain != question else None
    return found or _detect(question)


_LEAD = re.compile(r"^(ممكن|لو سمحت|من فضلك|معلش|طيب|عايزك|عاوزك|محتاجك|نفسي|ياريت|"
                   r"please|can you|could you|would you|i want you to|hey)\s+")
_TAIL = re.compile(r"\s+(لو سمحت|من فضلك|بليز|يا seshat|يا صاحبي|please|thanks)$")
_VERB_ME = re.compile(r"^(اكتب|الف|احكي|قول|اشرح|كلم|عرف|اعمل)(لي|لى|ني)\s+")


def _plain(question):
    q = question
    for _ in range(2):
        q = _TAIL.sub("", _LEAD.sub("", q))
    q = re.sub(r"^ت(كتب|حكي|شرح|كلم|عمل)(لي|لى|ني)?\s+", lambda m: "ا" + m.group(1) + " ", q)
    q = _VERB_ME.sub(lambda m: m.group(1) + " ", q)
    return q.replace("اكتب ليا ", "اكتب ").replace(" for me ", " ").strip()


def _detect(question):
    for kind, _, _, rx in _DETECT:
        m = rx.match(question)
        if m:
            topic = _TIME_TAIL.sub("", m.group(1).strip()).strip() or m.group(1).strip()
            if kind == "person" and (len(topic.split()) > 4 or topic.split()[0] in _NOT_NAME):
                return None        # "مين اخترع التليفون" سؤال عن معلومة، مش عن شخص اسمه كده
            return kind, topic
    return None


_NOT_NAME = set("""اخترع اكتشف كتب الف بنى بنا اسس عمل صنع رسم قال فاز كسب هيكسب هيفوز حكم قتل
هو هي اللي اول اخر اكبر اصغر اشهر احسن افضل اغنى اطول اسرع صاحب مؤلف مخترع مكتشف رئيس ملك
invented discovered wrote built founded made created painted said won wins killed ruled
is was are first last best richest tallest fastest will would can owns runs""".split())


# "سعر الدهب النهارده كام" → الموضوع "الدهب" مش "الدهب النهارده"
_TIME_TAIL = re.compile(r"\s+(النهارده|انهارده|دلوقتي|دلوقت|اليوم|حاليا|الان|today|now|right now|currently)$")


def _clean(topic):
    """الموضوع بيتكتب زي ما هيظهر في السؤال المتنضف (ة → ه، أ → ا…) عشان النسخ يبقى حرف بحرف"""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tokenizer import normalize_question
    return normalize_question(topic)


def _items(kind, topics, arabic):
    k = KINDS[kind]
    templates = k[3] if arabic else k[4]
    topics = [_clean(t) for t in topics]
    return [(kind, topic, arabic, [t.replace("{x}", topic) for t in templates]) for topic in topics]


def build(held_out=False):
    """
    بيرجع قايمة (النوع، الموضوع، عربي؟، الأسئلة، في_الذاكرة؟).
    held_out=True → مواضيع الاختبار بس (النموذج مبيتدربش عليها).
    """
    import unknown_questions as uq
    import wordlists
    rng = random.Random(2027)
    out = []

    test_only = {_clean(t) for t in HELD_OUT_AR + HELD_OUT_EN + HELD_OUT_AR_PEOPLE
                 + HELD_OUT_EN_PEOPLE + HELD_OUT_AR_TASKS + HELD_OUT_EN_TASKS}

    def add(kind, topics, arabic, memory=True):
        for item in _items(kind, topics, arabic):
            if held_out or item[1] not in test_only:        # مواضيع الاختبار متدخلش التدريب
                out.append(item + (memory,))

    if held_out:
        for kind in ("opinion", "price", "news", "about", "write"):
            add(kind, HELD_OUT_AR, True)
            add(kind, HELD_OUT_EN, False)
        add("person", HELD_OUT_AR_PEOPLE, True)
        add("person", HELD_OUT_EN_PEOPLE, False)
        add("howto", HELD_OUT_AR_TASKS, True)
        add("howto", HELD_OUT_EN_TASKS, False)
        return out

    banned = set(HELD_OUT_AR + HELD_OUT_EN + HELD_OUT_AR_PEOPLE + HELD_OUT_EN_PEOPLE
                 + HELD_OUT_AR_TASKS + HELD_OUT_EN_TASKS)
    ar_words = ["ال" + ar for _, ar in wordlists.WORDS if " " not in ar]
    en_words = [en for en, _ in wordlists.WORDS]
    ar_topics = [t for t in uq.AR_TOPICS if t not in banned]
    en_topics = [t for t in uq.EN_TOPICS if t not in banned]
    ar_people = uq.AR_PEOPLE + [f"{a} {b}" for a in AR_FIRST for b in rng.sample(AR_LAST, 3)]
    en_people = uq.EN_PEOPLE + [f"{a} {b}" for a in EN_FIRST for b in rng.sample(EN_LAST, 3)]

    for arabic, topics, extra, words, people, tasks, code, things, letters in (
            (True, ar_topics, AR_EXTRA_TOPICS, ar_words, ar_people, uq.AR_TASKS + AR_EXTRA_TASKS,
             uq.AR_CODE_TASKS, AR_THINGS, AR_LETTERS),
            (False, en_topics, EN_EXTRA_TOPICS, en_words, en_people, uq.EN_TASKS + EN_EXTRA_TASKS,
             uq.EN_CODE_TASKS, EN_THINGS, EN_LETTERS)):
        for kind, n_words, n_random in (("opinion", 45, 30), ("price", 45, 25), ("news", 30, 25),
                                        ("about", 0, 35), ("write", 40, 25)):
            add(kind, topics, arabic)                                   # المواضيع الأساسية → في الذاكرة
            add(kind, [t for t in extra if t not in banned], arabic, memory=False)
            add(kind, rng.sample(words, n_words), arabic, memory=False)
            add(kind, _letters_words(rng, letters, n_random), arabic, memory=False)
        add("person", people[:20], arabic)
        add("person", people[20:], arabic, memory=False)
        add("person", _letters_words(rng, letters, 25), arabic, memory=False)
        add("howto", tasks, arabic)
        add("code", code, arabic)
        add("best", things, arabic)
        add("best", _letters_words(rng, letters, 12), arabic, memory=False)

    for key, _, _, _, _, ar_qs, en_qs, _, _ in FIXED_KINDS:
        if ar_qs:
            out.append((key, "", True, list(ar_qs), True))
            out.append((key, "", False, list(en_qs), True))
    ar_junk, en_junk = uq.gibberish()
    out.append(("unclear", "", True, ar_junk, True))
    out.append(("unclear", "", False, en_junk, True))
    return out


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    items = build()
    print(len(items), "رد للتدريب |", sum(len(i[3]) for i in items), "سؤال |",
          len(build(True)), "رد للاختبار")
    for kind, topic, arabic, qs, _ in random.Random(1).sample(items, 6):
        print(f"  {qs[0]!r}\n     {think(kind, topic, arabic)}\n     {compose(kind, topic, arabic)}")
    print(detect("ايه رايك في القهوه"), detect("who is marie curie"), detect("عامل ايه"))
