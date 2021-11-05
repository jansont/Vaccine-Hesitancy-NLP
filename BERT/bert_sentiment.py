# -*- coding: utf-8 -*-
"""BERT_sentiment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1njz9gxVf1eQu2lQbMb5vU2oTyo5oGw1m
"""

pip install transformers

"""##Loading Pretrained BERT from TensorFlow"""

# Requirements: install HuggingFace Transformer library beforehand
# pip install transformers

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import pandas as pd

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model.summary()

"""##Dataset"""

from sklearn.utils import shuffle
df = pd.read_csv('dataset_bert.csv', names = ['LABEL_COLUMN', 'ID', 'D', 'Q', 'U', 'DATA_COLUMN'], 
                encoding = "ISO-8859-1",  error_bad_lines=False)
df = df[['DATA_COLUMN','LABEL_COLUMN']]

df.LABEL_COLUMN[df.LABEL_COLUMN ==4] = 1 

df = shuffle(df)
df = df[df['LABEL_COLUMN'].notna()]
df = df[df['DATA_COLUMN'].notna()]

train = df.iloc[0:250000]
train.head()

test = df.iloc[250000:260000]
test.head()

"""Next, we need to format the data such that it is recognised by the TFBertForSequnceClassification model. 

The pretrained BERT model takes three input features:

— **input ids**: Input ids are an id number assigned to each word based on the existing BERT vocabularies. 

— **token type ids**: Since BERT tokenizer helps you to pad your sentence with 0 so that every sentence is of the same length, token type ids are required to differentiate between actual words and paddings. 

— **attention masks**: Attention masks help to recognise which sentence does the word belong to.

 BERT tokenizer has a function encode_plus which converts your raw sentences into the three input features. The following code helps to organise your dataset in Tensors, such that it is compatible with BERT tensorflow implementation.

##Transform Data for BERT
"""

def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN): 
  train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)

  validation_InputExamples = test.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)
  
  return train_InputExamples, validation_InputExamples

  train_InputExamples, validation_InputExamples = convert_data_to_examples(train, 
                                                                           test, 
                                                                           'DATA_COLUMN', 
                                                                           'LABEL_COLUMN')
  
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for padding
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


DATA_COLUMN = 'DATA_COLUMN'
LABEL_COLUMN = 'LABEL_COLUMN'

# tokenizer.encode_plus()

# train and test is your dataset
train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

# Setting up callbacks for TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.fit(train_data, 
          epochs=2, 
          validation_data=validation_data,
          callbacks=[tensorboard_callback])

# optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)

# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# model.fit(train_data, epochs=1, validation_data=validation_data)

pred_sentences1 = ["hey", "Earlier this month, HP began selling the Spectre, the worlds thinnest laptop, according to the company. At 0.41 inch thin, its as flat as a single breakfast pancake—bananas not included. More impressive, it doesnt skimp on processing power, like [Apples new MacBook] does. [HP executives have focused] their efforts on out-innovating the premium laptop maker. Yes, the MacBook is 0.1 inch fatter than the Windows 10-powered Spectre. To the naked eye, that means nothing. Even with a 5X magnifying glass and tape measure, I could barely spot the difference. Its why Ive long felt technologys thinolympics has been a waste of time. Our new product is so much thinner than the competition that you can fit one more sheet of paper into your messenger bag! Youll need to make room for the charger, though, since we cut out some battery. Sorry! The equation has long been: Thinner + lighter = poorer performance + shorter battery life. Both the Spectre and the MacBook, [updated in April], still require you to make some sacrifices. But the trade-offs no longer outweigh the benefits of owning a laptop that could double as a cheese knife—if thats what you want. All-Important Ports Ports are to laptops what ice cream is to people. Limit them, and youll be shocked by the Before and After shots. Other than the headphone jack, the MacBook has just one USB port—and its not even the kind youre familiar with. The size of a Tic Tac, the [USB Type-C port] is used to charge the laptop and attach peripherals, including an external display. Well get over the limited port situation just like we got over the disappearance of DVD drives. (Remember those?) But in the here and now, there are two issues. First, since the port is uncommon, you cant do things you take for granted, like plugging an iPhone into your computer with the regular USB cable. Second, one port just wasnt enough for me. The best solution? Purchase a hub with full-size USB ports and an SD card slot, like the [$50 Hyperdrive USB Type-C 5-in-1 Hub]. Apple should take a page from HPs port playbook: The Spectre has three USB Type-C ports along its back, two of which are capable of handling more power than the MacBooks. And if you buy the Spectre from Best Buy, a USB-Type-C-to-regular-USB adapter is included in the box. The Key(s) to Comfort The MacBook and the Spectre represent the laptop stripped to its barest parts: screen and keyboard. Yet theyre not like the [tablet-keyboard combos], which fail at the whole sitting-on-your-lap thing. The traditional clamshell design makes these a pleasure to use—if you can get used to a few ergonomic shortcomings of their own. The MacBooks sturdier, more attractive build is tarnished by a keyboard that looks like its been flattened by a dough roller. To make the underside razor-thin, the keys were chopped off and redesigned with a mechanism that aims to recreate the feedback and bounce of real keys. It took some getting used to, but three months in, my hands were so comfortable, it felt a little weird going back to the MacBook Air. From the start, I loved the MacBooks large trackpad. Though its just a flat, fixed piece of glass, small vibrations and a clicking sound fool you into thinking youre physically pressing down. It messes with your mind—in a good way. With the Spectre I felt at home on the keyboard in an instant. The keys are a normal height, with more surface area thanks to a 13.3-inch screen (versus the MacBooks 12-inch display). HPs trackpad, on the other hand, feels claustrophobic. I also repeatedly encountered issues with it, including jumping cursors and unregistered clicks. HP says it is working on a software update to fix the problems. The unfixable issue with the Spectre? The fact that it looks like it was designed by Kanye West. The only color option includes a gaudy gold logo and hinge that instantly attract fingerprints. On the other hand, I am entranced by HPs trippy new logo. Its like a Magic Eye optical illusion. Does it say HP? Lip? Fiji? Prioritizing Performance Since thinner laptops have less room for battery, they tend to have less powerful chips. Yet the newest chips found in the MacBook and Spectre amp up the processing while remaining relatively efficient. Apple recently updated the MacBook with the new Intel Core M processor, which is 20% faster than last years chip. I noticed it. The previous model took too long to open apps and multitask. The new ones snappier at my usual routine of juggling multiple browser tabs and apps like Spotify, Microsoft Word and Slack. Slowdowns only start occurring when I throw more graphics-intensive jobs at it in Photoshop. ([Ive changed my mind.] It could now potentially replace my three-year-old MacBook Air, though Id like to see whats next for the MacBook Pro before deciding.) The Spectre is available with Intels big-boy Core i5 and i7 processors. These can provide up to 25% more power than the Core M processors. In my tests, the Spectre was just as snappy as the MacBook at surfing the web and launching apps, but when it came to editing multiple, large images in Photoshop, the HP was far more cooperative. That speed has some downsides. The Spectres fan periodically sounded like it was preparing to cool down an office building. A software fix quieted it down a bit, though it still acts up at times. (The MacBook doesnt have a fan, and suggests closing programs when it gets warm.) The Spectre also trails behind the MacBook on battery. In my test, which loops a series of websites with brightness set at around 80%, the Spectre ran for 6.5 hours. The MacBook lasted for 8.5 hours, an hour beyond last years model. And it pulls that off while driving over a million more display pixels than the HP. In daily use, I found the MacBook still outlasted the Spectre, though both fell short of the 11-hour endurance of the larger MacBook Air and Dell XPS 13. (If battery life concerns you, try using [native browsers—Edge on Windows, Safari on Mac].) The Spectre and the MacBook are symbols of computing progress, and good news for people shopping for Windows or Mac hardware. Just look how much the MacBooks performance and battery life improved in one year. Still, before you are seduced by thinness, ask yourself how much you value portability over ports, performance and battery life—and how much youd pay for the compromise. The Spectre starts at $1,170 and the MacBook at $1,300. Meanwhile, for under $1,000, you can buy one of their, uh, huskier counterparts. Write to Joanna Stern at [joanna.stern@wsj.com] or on Twitter [@joannastern]"]
pred_sentences2 = ["Koca is under increasing attacks for not fully disclosing the number of people who test positive for the coronavirus, much like the rest of the world does. Turkey’s opposition parties and medical associations criticized the minister for causing negligence by portraying a rosy outlook and the government for prioritizing economic gains over the lives of people. In Istanbul alone -- Turkey’s largest city which Koca said on Nov. 2 accounted for 40% of the national patient count -- an average of 409 people died every day during the past week, a 108% rise from the same period a year ago. Mayor Ekrem Imamoglu, an opposition heavyweight, has frequently pointed out the discrepancy between data on Istanbul fatalities and the comparably low number of reported coronavirus deaths nationwide. He has recently urged the government to impose a full lockdown to contain the spread of the virus, a call endorsed by Meral Aksener, the leader of the nationalist Iyi Parti."]
pred_sentences3 = ["Turkey announced a record number of deaths from the coronavirus, highlighting the dilemma facing policy makers trying to contain the current surge in new cases without shutting down the economy again. The Ministry of Health on Monday reported 153 deaths due to the virus over the past 24 hours and announced 6,713 symptomatic patients, bringing the total number of reported cases to over 453,000 since the outbreak began nine months ago. The number of new cases reported over the past day doubled from a week ago, a pace unseen since the early stages of the pandemic when the increase was often exponential. Equally alarming is the fact that Turkey made a controversial tweak to its data reporting in July, excluding asymptomatic cases and reporting only symptomatic patients. The government can hardly afford another lockdown with mounting costs from a contraction in activity earlier this year, rising public expenses to support job programs and the fallout on tax revenues."]

pred_sentences = ['This was an awesome movie. I watch it twice my time watching this beautiful movie if I have known it was this good',
                  'One of the worst movies of all time. I cannot believe I wasted two hours of my life for this movie']



input_sentences = pred_sentences2

tf_batch = tokenizer(input_sentences, max_length=128, padding=True, truncation=True, return_tensors='tf')
tf_outputs = model(tf_batch)
tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
labels = ['Negative','Positive']
label = tf.argmax(tf_predictions, axis=1)
label = label.numpy()
for i in range(len(input_sentences)):
  print(input_sentences[i], ": \n", labels[label[i]])
