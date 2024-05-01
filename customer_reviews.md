## Instaling the required packages


```python
import os
import time
import json
import requests
import tiktoken
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, dotenv_values
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_community.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain

load_dotenv()
```




    True



#### Decide about loading new data


```python
# importing inputs from the UI 
from user_inputs import new_data_st, max_cus, product
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    Cell In[111], line 2
          1 # importing inputs from the UI 
    ----> 2 from user_inputs import new_data_st, max_cus, product
    

    ImportError: cannot import name 'max_cus' from 'user_inputs' (C:\Users\Mast_Nijanand\customer_review_app\user_inputs.py)


#### Setting up small vs. large content code toggle


```python
if new_data_st == "Yes":
    new_data = True
else:
    new_data = False
```


```python
# Setting up logical code block execution toggle
class StopExecution(Exception):
    def _render_traceback_(self):
        pass
```

#### Getting API Keys


```python
# Activating the API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
X_RapidAPI_Key = os.environ.get("X_RapidAPI_Key")
Rapid_AI_URL = os.environ.get("Rapid_AI_URL")
Rapid_AI_Host = os.environ.get("Rapid_AI_Host")
```

#### Invoking Amazon Scrapper


```python
class AmazonScraper:
    __wait_time = 0.5

    __amazon_search_url = 'https://www.amazon.com/s?k='
    __amazon_review_url = 'https://www.amazon.com/product-reviews/'

    __star_page_suffix = {
        5: '/ref=cm_cr_unknown?filterByStar=five_star&pageNumber=',
        4: '/ref=cm_cr_unknown?filterByStar=four_star&pageNumber=',
        3: '/ref=cm_cr_unknown?filterByStar=three_star&pageNumber=',
        2: '/ref=cm_cr_unknown?filterByStar=two_star&pageNumber=',
        1: '/ref=cm_cr_unknown?filterByStar=one_star&pageNumber=',
    }

    def __init__(self):
        pass

    def __get_amazon_search_page(self, search_query: str):
        # setting up a headless web driver to get search query
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        driver = webdriver.Chrome(options=options)

        url = AmazonScraper.__amazon_search_url + '+'.join(search_query.split())
        driver.get(url)
        driver.implicitly_wait(AmazonScraper.__wait_time)
            
        html_page = driver.page_source
        driver.quit()

        return html_page

    def __get_closest_product_asin(self, html_page: str):
        soup = BeautifulSoup(html_page, 'lxml')

        # data-asin grabs products, while data-avar filters out sponsored ads
        listings = soup.findAll('div', attrs={'data-asin': True, 'data-avar': False})

        asin_values = [single_listing['data-asin'] for single_listing in listings if len(single_listing['data-asin']) != 0]

        assert len(asin_values) > 0

        return asin_values[0]

    def __get_rated_reviews(self, url: str):
        # setting up a headless web driver to get search query
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        driver = webdriver.Chrome(options=options)

        driver.get(url)
        driver.implicitly_wait(AmazonScraper.__wait_time)

        html_page = driver.page_source
        driver.quit()

        soup = BeautifulSoup(html_page, 'lxml')
        html_reviews = soup.findAll('div', attrs={"data-hook": "review"})

        reviews = []
        # extract text from various span tags and clean up newlines in their strings
        for html_review in html_reviews:
            name = html_review.find('span', class_='a-profile-name').text.strip()    

            # Amazon's format is "x.0 stars out of 5" where x = # of stars
            rating = html_review.find('span', class_='a-icon-alt').text.strip()[0]

            review_body = html_review.find('span', attrs={'data-hook': 'review-body'}).text.strip()

            reviews.append({'customer_name': name, 'rating': int(rating),'review': review_body})

        return reviews

    def __get_reviews(self, asin: str, num_reviews: int):
        if num_reviews % 5 != 0:
            raise ValueError(f"num_reviews parameter provided, {num_reviews}, is not divisible by 5")

        base_url = AmazonScraper.__amazon_review_url + asin
        overall_reviews = []

        for star_num in range(1, 6):
            url = base_url + AmazonScraper.__star_page_suffix[star_num]

            page_number = 1
            reviews = []
            reviews_per_star = int(num_reviews / 5)

            while len(reviews) <= reviews_per_star:
                page_url = url + str(page_number)

                # no reviews means we've exhausted all reviews
                page_reviews = self.__get_rated_reviews(page_url)

                if len(page_reviews) == 0:
                    break

                reviews += page_reviews
                page_number += 1

            # shave off extra reviews coming from the last page
            reviews = reviews[:reviews_per_star]
            overall_reviews += reviews

        return overall_reviews

    def get_closest_product_reviews(self, search_query, num_reviews, debug=False):
        if debug:
            start = time.time()

        html_page = self.__get_amazon_search_page(search_query)
        product_asin = self.__get_closest_product_asin(html_page)
        reviews = self.__get_reviews(asin = product_asin, num_reviews = num_reviews)

        if debug:
            end = time.time()
            print(f"{round(end - start, 2)} seconds taken")

        return reviews
```

## New Product Selection

### Getting Amazon Reivews


```python
if new_data:
    search_query = str(product)   # 'premier protein shake, chocolate'
    scraper = AmazonScraper()
    reviews = scraper.get_closest_product_reviews(search_query, num_reviews = max_cus, debug=True)
    print(reviews)
```

    208.15 seconds taken
    [{'customer_name': 'The Living Waters Farm', 'rating': 1, 'review': "This phone is glitchy brand new out of the box...  When I try to scroll the screen jumps around and when I try to type it adds characters that I did not touch...This was advertised as a brand new phone but I think they gave me a remanufactured one with brand new screens put over top of a phone that had internal damage or something...The Amazon store is called the Samsung store so I thought I was getting a device directly from Samsung but it turns out that the seller is actually called technomaster and it's just using the Samsung name...Beware, just figure something else out and go through your cell phone company so that you will get a reliable device not a remanufactured polished turd..."}, {'customer_name': 'Nyron C.', 'rating': 1, 'review': "We bought this phone July of last year.  We paid just over a thousand dollars for the phone. Here we are I'm March and the phone is not taking calls,  it goes directly to voice mail. Finally, the phone company advised us that someone  puts a block on the IMEI chip/ number on the phone. We reached out to Samsung and they sent us to Amazon  which we connected with vendor  who told us we must have downloaded something ti block the phone.  I'd that's the case everyone's phone  would pretty much be blocked.  We were advised  that  this is something someone  who has access to the phone can do. Now  we're put of over one thousand dollars , no phone  and only 8 months since we bought this phone!"}, {'customer_name': 'Jesse F', 'rating': 1, 'review': 'Ordered this and I was sent the Singapore version of the Samsung phone. This isn\'t some US unlocked international version. This requires you install an app made by the Government of Singapore. Also overall the build quality felt poor. The side panels didn\'t line up well with the glass. It felt sharp in my hands, like some low budget version that Samsung has made in Vietnam (where this model was made). I bought one of these "international" versions of another phone like that and had endless trouble with support and getting 5g to work. STAY AWAY unless you are looking on sending data to a foreign government and dealing with a host of incompatibilities with 0 support.'}, {'customer_name': 'Brett C.', 'rating': 1, 'review': "I bought my first one, it immediately had blinking screen , battery lasted about 6 hours (barely any use), phone calls distorted, echoing, restarted many times no support except call Samsung. Appparently it's a new device and told to return it to this place.I took 8 days to credit my account.  Amazon won't give me my money back.  I had to order another device from this place.  Again, same issues but worse.  Had to call Samsung.  Send my phone to an authorized dealer which will take a week, and then who knows.I have to buy a temporary phone just to get my new device fixed.BTW- the issues happen immediately when starting up the phones- yes, Plural.As it stands, I cant get my money back from Amazon."}, {'customer_name': 'Maksym Krokhmalov', 'rating': 1, 'review': "I decided to save money and take used (like new). This is a lie. The phone is in poor condition, I didn't even turn it on. The entire screen has small scratches, the corner of the phone is scratched and slightly wrinkled, as well as several other small scratches and chips on the body. Phone without original box. I'm disappointed and will take a new one; I won't take any more risks with used ones."}, {'customer_name': 'Happy member', 'rating': 1, 'review': 'First of all it is new. But it came in a box with language I couldn’t recognize. Ok, unboxed and turned on. And you know what? It is preset language from factory. I struggled to find how to change to English. Then i tried to activate ESIM and add to my Verizon account but couldn’t. Next day i went to store and they couldn’t either. Then we put physical simcarf and it worked somehow but only on 4g. After all those struggles i returned this odd samsung.'}, {'customer_name': 'Jamal tarhrouti', 'rating': 1, 'review': "I don't recommend this one to anyone it losing the signal I try different phone carriers still the same problem I try with at&t has exactly signal  but with phone don't work"}, {'customer_name': 'mrt685', 'rating': 1, 'review': 'Phone lasted 2months before started having problems. Touch screen playing up aswell as back light flickering. Has progressively gotten worse to the point where I can no longer use. Have been in contact with the seller and they just tell me to get in contact with Samsung customer service. Not happy at all with the amount I spent on the phone for what I got, especially the service from the seller. The seller stopped replying to my messages. Will retract my statement if seller actually replies and does something about it.'}, {'customer_name': 'Maddyson T.', 'rating': 1, 'review': 'If I could, I would rate it a 0 star. This is a scam, it says its "globally unlocked" but it is not. The card that comes with the phone says it works for AT&T and all those other carriers but they do not work at all. Amazon needs to do something about this because it\'s giving us customers false advertisements. Very disappointed on this purchase. All this money us customers spent is gone, all because of false advertisements.'}, {'customer_name': 'dannyboy30', 'rating': 1, 'review': 'loved the phone fast shiping but the battery lasted very short time like 3 to 4hrs and phone acted up alot as a boat captain i can not have that so i returned it they keep 90 dollers on restocking fee on a mest up item ☹️😢'}, {'customer_name': 'Eric', 'rating': 1, 'review': "Received defective or improperly setup phone. Voice commands and queries do not work on any app or camera, even though microphone has permission for each app. Google Assistant does not work with voice commands. Looked at lots of videos but no suggestions helped. Reached out to seller several times but no reply.  No customer service on a $1000 phone?  Aside from the voice commands not working, this phone was not setup for the U.S. market. First, you can't shut it off, as the on/off side button has been set to Bixby. You'll need to fix that in your settings. Next, the language is UK English, Not American. The language needs to be changed not just in one place, but changed on many apps. First phone I've ever had to return."}, {'customer_name': 'MM', 'rating': 1, 'review': 'Cuidado, "Cell Universal" la empresa que lo comercializa, no proporciona factura de los productos que vende. En mi caso la solicité a través de Amazon y directamente a Cell Universal y no te llega la factura; Amazon responde que esta empresa ya no opera en Amazon. Ha sido mi experiencia. Compras un producto de alta gama y no te emiten la factura, solo el recibo. Es sospechoso.'}, {'customer_name': 'Stephen', 'rating': 1, 'review': 'As of August 2023 this phone IS NOT, I emphasize NOT compatible with Spectrum in the USA for wifi calling.  Do not even bother trying to get technical support on this issue from either Spectrum or Samsung.  They each blame each other and all you will get is canned responses from the technicians.Purchase this phone at your own risk and MAKE CERTAIN all features you want operate with your carrier, before you get out of the return window.After a two month runaround from both companies, that window closed for me and now I am apparently screwed.'}, {'customer_name': 'Anonymous', 'rating': 1, 'review': 'Do not get'}, {'customer_name': 'Luminita Pamint', 'rating': 1, 'review': 'Dissatisfied'}, {'customer_name': 'Robert Kenneth Perlstein', 'rating': 1, 'review': "MMS arrives compressed to 40kb max, down from 3MB. This means all images are incomprehensible and video messages cannot be received at all.Verizon clueless as how to help. I'd advise against this phone if you are on Verizon."}, {'customer_name': 'george', 'rating': 1, 'review': 'Seller claimed this is factory unlocked for all carriers. When i got it it came with a paper listing the carriers it is not unlocked for. And at&t and cricket are not supported.. the only way to return it is to have ups pick it up from my house and i work the next week.'}, {'customer_name': 'Debbie C.', 'rating': 1, 'review': 'I did like this phone except I could not watch videos by text. Also pictures received by text were blurry. Neither Verizon or Samsung could fix it so the phone had to be returned. Back to using old phone with no problems other than I can no longer get updates.'}, {'customer_name': 'Jeanne', 'rating': 1, 'review': 'Disappointed'}, {'customer_name': 'Amanda', 'rating': 1, 'review': "Can't receive videos in text messages on Verizon.  Wish I would have read the reviews before buying. I threw out the box so probably can't return it."}, {'customer_name': 'Veljko Miljanic', 'rating': 1, 'review': 'This phone is complete crap. You will give $1000 dollars for a phone and you can\'t use any USB-C headphones. When you connect them phone just writes "unsupported USB device". Basically, they want to lock you in into baying only Samsung stuff to overcharge for low tech products that anyone can produce.I will never buy Samsung phone again.'}, {'customer_name': 'Amazon Customer', 'rating': 1, 'review': "The Samsung phone I received from this seller is meant for UK wireless networks.  It will receive some texts but others will be blocked because it is not recognized as a US phone. Will be sending back. Also. Uhm. The only way to send it back is to schedule a pick up.  I've tried 3 times waited multiple days.  No one comes. I think this is a scam so you never get your money back!"}, {'customer_name': 'Etien Valdes Luzardo', 'rating': 1, 'review': "The phone is good if you don't care about 5G speed. I had to replace 3 Sim cards because the phone was blowing them up after getting 5G, 4G and LTE works good"}, {'customer_name': 'Alexander B.', 'rating': 1, 'review': "It seems like the phone is not new. The sticker has been resealed (see the photo). I haven't even opened the packaging. The phone inside seems to be loose, as if it's not secured in the box."}, {'customer_name': 'Saravanan', 'rating': 1, 'review': 'This phone is bad in wifi connection not pulling good as my old phone'}, {'customer_name': 'DSulkey', 'rating': 1, 'review': 'Couldn’t make Internet calls. Foreign phone'}, {'customer_name': 'Karina M', 'rating': 1, 'review': 'False Advertisement'}, {'customer_name': 'Clair Wiederholt', 'rating': 1, 'review': 'Amazon is deducting a $132.60 re-stocking fee on this defective product.  AVOID SHOPPING AT AMAZON.  Walmart has better service.'}, {'customer_name': 'Sharon', 'rating': 1, 'review': 'My phone is defective. Screen flickers, slow speed, issues opening apps that I previously had no problem with on my other phone.'}, {'customer_name': 'James Moses', 'rating': 1, 'review': 'The ad says unlocked for all carriers. This phone will not work with cricket wireless.'}, {'customer_name': 'Onerazorsharp', 'rating': 2, 'review': "This was a good deal for the price point, and one of the few reputable places to get the green S23 ultra.  With the s24 issues, I wanted to replace my s23 ultra with the same.  Asurion is back-ordered so far they offered to pay me cash value.  So this is the direction I went.The problems weren't apparent until I tried to cut the SIM and LTE service over.  What you'll find with this device in the package is a card telling you the APN settings you'll need to apply.  This is where issues start however, that you couldn't save/apply the settings you were given - the apn registry will just disappear and you'd be left with CDMA apn only.  To keep the explanation simple I spent hours researching and playing with APN settings to no avail.  Voice service worked with the default APN settings that were registered to the SIM card, but SMS and MMS wouldn't.  Well in today's world of 2 factor Auth that's not good - especially since I've already registered most of 2FA to use SMS to my number.So I go to a Verizon store and spend 4 hours before the Superbowl trying to figure it out, adding new SIM(s)/ESIM(s) and nothing will get SMS to work.  I then realize that my watch LTE is not working either and we can't activate that due to the line sharing.  So we deactivate my line and attempt to re-activate each separately, and now my watch is the only one with service at all.  VZ tier 3 figures out a way to remove my account entirely from VZ network and re-add, and then I have a working watch LTE and phone except it's SMS only.  No MMS, no texts working to iPhone users, and super spotty coverage.  Eventually VZ decides to do a CLNR (certified like new replacement) to me for all this trouble.  BAM, this one works immediately with 0 issues.  Something about this global model and the service bands it uses caused serious troubles for me.If you're looking for a phone that handles basic SMS and LTE, then this does ok.  You just won't get mms to work domestic USA.  Also, Dual SIM won't work in US to run 2 lines, but you should be able to activate the second SIM on other country's providers."}, {'customer_name': 'Frankybigguns', 'rating': 2, 'review': "This phone is PERFECT if you plan to move overseas, however if you intend to use it in the US keep in mind that this phone will not be covered under Samsung's USA warranty. Furthermore, there are a number of networks that will NOT work with this phone such as Cricket, US wireless, Boost mobile, and a few others. Because of this, my phone received a very weak signal and I could barely hear or talk to any one I dialed. I ended up returning this and am still waiting for my refund and decided I would just go with official phones sold from Samsung, Google, and/or Best Buy.EDIT: Seller refunded me about a week after sending back. I wanted to add that the phone functioned just fine outside of cell signal reception and WIFI calling functioned flawlessly. However due to the region lock which I could not swap APNs on (including downloading the software), the Singaporean government app installed itself onto the phone for which I could not delete (I could delete it but it would come right back). Added 1 star because the phone functions fine and the seller was gracious in returns, but being unable to unlock the region/change APNs and the limitations in US carriers keeps my overall rating below average. Your mileage will vary..."}, {'customer_name': 'Abe', 'rating': 2, 'review': 'Iove this phone so far, but beware you will not be eligible for Samsung care+. This was very irritating as this was never disclosed anywhere on the listing. All it took was a 3" drop on to a smooth surface to crack the entire bottom half of the screen. Now I\'m going to be paying an additional $350 for a screen repair on top of the $900 I already paid for the device. Thanks a lot universal goods. 🖕'}, {'customer_name': 'Colten', 'rating': 2, 'review': 'I have been using Samsung for a long time and I have never wanted to break one of their phones more than this one.  If I take a picture and then decide to send it to some one, it is hit or miss if it will be there. If not I have to close everything down and then it is usually there.  If I send a video, it can only be a second long.... YES 1 SECOND long.  I read about this, some people say it just fixes itself.  When someone sends a picture to me, hit or miss if it is a visible.  Sometimes it is all scrambbly.  I will look at the same pic on my wifes phone from the same person and it is fine.  If I send a GIF in a text, it ALWAYS says this may not send because it might be too big.  I always sends but why should a message even come up like that when sending a GIF?  HATE THIS PHONE!!'}, {'customer_name': 'Miguel', 'rating': 2, 'review': 'The product was sold as new, it was not. There is no mention on the description of the phone being from Singapore.DOES NOT WORK WITH VERIZON 5G NETWORK.If you are bying this phone for only talk and text, it works, other wise looks elsewhere.'}, {'customer_name': 'D. Deluca', 'rating': 2, 'review': 'Nothing but issues with this phone from day one. Wish I never got rid of my S20FE.'}, {'customer_name': 'Corbin Allen', 'rating': 2, 'review': 'Not globally unlocked for all carriers in the US. Spectrum, Boost, cricket, or US Cellular. It came with a notecard informing me of this and I verified with the IMEI#. Unfortunately I had to return it.'}, {'customer_name': 'Jerry', 'rating': 2, 'review': "This product was advertised to support AT&T wireless in US.  But it won't work for AT&T wireless.  In addition, it charged $132 restocking fee for returning item.  Tried to contact them without any success and the same for Amazon support."}, {'customer_name': 'Henry Aguilar', 'rating': 2, 'review': 'El equipos lo recibo en perfectas condiciones .. sin embargo en los últimos días.. la batería su carga de 100% no dura más de 6 horas ..por momentos hay un parpadeo en la pantalla que no me explico por qué pasa.  🤔.  Pero necesito el teléfono por que hace parte de mi trabajo y lo necesito a diario ..'}, {'customer_name': 'Thomas C', 'rating': 2, 'review': "Reception on this phone is pitiful. Every phone I've had up till this one was good to go. I have constant issues with signal and wifi. Just recently I couldn't even make a messenger video call. I wish I had my s10+ instead of this piece of crap. Light-years better"}, {'customer_name': 'Ookimaru Najami', 'rating': 2, 'review': 'El teléfono venía con pantalla rota y caja abierta. El teléfono no es americano si no es de Chino, de Singapore. Tomen eso en cuenta.'}, {'customer_name': 'Jules', 'rating': 3, 'review': 'This particular listing does not have the "may not work on 5g\' disclaimer that some of the other unlocked S23 Ultra listings have so I bought it.  Popped in my sim, no 5G.  I have a post paid plan on AT&T in the U.S.  I tried using the APN settings that came in the box, no dice.  Factor reset, waiting, ect.  No matter what I tried, I was still stuck on AT&T\'s LTE network.  As a sanity check I put the same SIM right back into my Pixel 7 Pro and was right back on 5G and 5G+ within seconds.Also, for some reason, the phone was region locked to Singapore.  So I was being forced to install "SGSecure."  Its some Singapore police suspicious activity reporting app, and I don\'t need or want that.Sent it back asap but still waiting for the refund weeks later'}, {'customer_name': 'Tabitha Wolfe', 'rating': 3, 'review': "I was super excited to get this phone. Unfortunately my cell provider didn't allow for this phone. After spending and hour trying to change the language and getting a SOVIET GOVERNMENT  message we decided to send it back as we would not be able to utilize the phone with our current provider. The phone was in great shape, but setup was an absolute nightmare due to all instructions being in a foreign language and the alarming government message. I've never seen something like that in my life."}, {'customer_name': 'Roger Goins', 'rating': 3, 'review': 'I thought Samsung made good products. Just not this one. Multiple issues.'}, {'customer_name': 'Daniel Martin', 'rating': 3, 'review': "I ordered the green color and received the black one, it doesn't affect the operation but I liked it in green (and I paid more for it (the black one was cheaper))"}, {'customer_name': 'Bill Keener', 'rating': 3, 'review': 'Some reviews said yes, it is Verizon compatible and some said it is not.Verizon said "It is not"'}, {'customer_name': 'Prasanna', 'rating': 3, 'review': 'Global Model, unlocked, but not working with Spectrum Mobile, maybe does not supports US Bands. Returning it.'}, {'customer_name': 'Aderaw', 'rating': 3, 'review': "Phone is good,  but the battery longevity starts decreasing even if I turn off background apps. I don't even use it frequently as I am busy mostly."}, {'customer_name': 'Donato perez', 'rating': 3, 'review': 'No es el color que ordene  muy mal.'}, {'customer_name': 'Tao', 'rating': 3, 'review': "I found scratches on the frame, so I think it's probably a used phone."}, {'customer_name': 'Vero', 'rating': 4, 'review': 'The box does not come closed from the factory, You can tell somebody opened it and changed the seal. It also makes you download a Singaporean government app. Other than that everything works pretty good.'}, {'customer_name': 'LigNosmas', 'rating': 4, 'review': 'Satisfied with the features, Android 14It is difficult to apply a screen protector due to the curved bezel.'}, {'customer_name': 'Amazon Customer', 'rating': 4, 'review': 'I have just started using the phone. My present location is in the Caribbean. It is functioning. I am using it to send this message. However, I believe because I use bluetooth with earplugs, the battery has to be charged ever so often.I prefer the fingerprint lock at the back for the S9. I do not like that the fingerpeint lock is on the face of the S-Ultra. I prefer the way the screenshots are done on the S9. To do screenshots on S-Ultra you have to press the volume up/down and power buttons, both located on the right side at the same time lightly. It took me some getting used to before I got it right. I am still familiarizing myself with the features on the S-Ultra. I will provide further updates as time progresses.'}, {'customer_name': 'Arlet Smith', 'rating': 4, 'review': 'It was an awesome buy.  Bought it for my BF and he loved it.'}, {'customer_name': 'Lee', 'rating': 4, 'review': 'The phone was easy to set up transferring from my S10 in few minutes and works perfectly in South Africa'}, {'customer_name': 'Kevin Solis', 'rating': 4, 'review': 'Had some trouble with shipping from seller and had to reorder for a higher price but ended up receiving item on time and the phone works good , brand new double sim and as expected.'}, {'customer_name': 'Spike-MD', 'rating': 4, 'review': "It is so much better than my old mobile, which was a Huawei.  It doesn't drain the juice quickly, no matter if I have it on for 12 hours.The Pro camera feature needs improvement though.  My pictures turn out blurry when I use it.  I'll just use the standard camera feature."}, {'customer_name': 'Opal', 'rating': 4, 'review': 'No issues. Nice phone.'}, {'customer_name': 'Scott', 'rating': 4, 'review': "This Samsung Galaxy S23 Ultra 5G (SM-S918B/DS) Dual SIM 256GB/ 8GB RAM, GSM Unlocked International Version does work on Verizon for phone calls, text message, and data.  Verizon will only allow WiFi Calling and Visual Voicemail on the US Version model SM-S918UZ. Verizon's Basic voicemail did work on the international version.  I had to return the internation model because where I live I need WiFi Calling."}, {'customer_name': 'Adiel de Jesús Alejandro Herrera', 'rating': 4, 'review': 'El teléfono es perfecto en todo, el único problema fue que yo lo pedí color dorado y me llegó de otro color'}, {'customer_name': 'Song', 'rating': 4, 'review': "The phone was great, but this specific seller's version does not support Cricket as a carrier. I had to return. Just informing other potential buyers to be aware of this."}, {'customer_name': 'prospero leonardo', 'rating': 4, 'review': 'Muy felis com mi tel.nuwvo samsung'}, {'customer_name': 'Tamesha Goddard', 'rating': 4, 'review': 'Used the product as a work phone and it works well thus far. Purchased it in April, delivery was seamless and set up the same. No issues encountered.'}, {'customer_name': 'Alvia', 'rating': 4, 'review': 'With this product,  I ordered the green color.  When I received the phone,  it was white.Other than the color,  the phone itself is good.  So far,  the battery life is long lasting.'}, {'customer_name': 'JHOOD', 'rating': 4, 'review': 'Product description states Green, got Black color.'}, {'customer_name': "INSTITUTION CHRETIENNE D'HAITI", 'rating': 4, 'review': 'Top phone'}, {'customer_name': 'Neniux😉', 'rating': 4, 'review': 'Funciona perfecto. No me gusta mucho como salen las selfies,  pero lo demás excelente'}, {'customer_name': 'Tumbilah', 'rating': 5, 'review': 'Camera quality is amazingCharges wellGood storageDual SIMValue for money'}, {'customer_name': 'Kayannaphora', 'rating': 5, 'review': 'I really wanted the S23 Ultra because I like the curved edges, which my old phone also had, but no phone carrier or store had them in stock. I was nervous about ordering a phone off Amazon, but my order came fast, in the factory packaging, with no issues and a nice little gift (a charger) from the seller. Plus, the phone is everything I wanted: battery lasts for days, tons of storage, simple to transfer data from my old Galaxy, and charges quickly. All I had to do to set it up was transfer the SIM card. Overall, great buy.'}, {'customer_name': 'Regina Daniels', 'rating': 5, 'review': 'I wrote a review earlier BUT I did not know a charger did not come with the cell phone.  Normally a SIM Card is included but it was not.  The small items that should be include are not.  My delivery read left in mailbox; "NO" the box (Too large for the mailbox) and my mail was left on the front porch.  Also, the charge is not included.  Basically, you\'re just getting the phone with no SIM Card and the basic Charger.The cell is a great phone.'}, {'customer_name': 'C Perez', 'rating': 5, 'review': 'Upon opening the package, I read the introduction card, which states "This phone is Not compatible with Cricket Wireless." So I returned it. I should have simply installed my Cricket Sim, and it probably would have worked fine like my daughters. Live and learn.'}, {'customer_name': 'Adrian P. Ross', 'rating': 5, 'review': 'I thought i commented regarding this phone already, neway, good phone the first day i powered this up it got a bit warm but since then its been about a month i dont have any complaints at all. Highly recommend'}, {'customer_name': 'maria e.', 'rating': 5, 'review': 'No trae cables necesario para la carga de batería'}, {'customer_name': 'Amazon Customer', 'rating': 5, 'review': 'Muy contento'}, {'customer_name': 'AnchorsAway', 'rating': 5, 'review': "After my husband screen broke and used our warranty to get a nothing one. Instead of upgrading we love this phone so much we keep with it. We haven't had any issues with this phone and would give this one 10 stars."}, {'customer_name': 'Raymond L', 'rating': 5, 'review': "After Having Iphone for almost 10 years I decided it was time for a change and haven't look back since. I love the S Pen everything about this phone there is only one down fall this phone doesn't run on 5G UC only on 5G. Other than that I love this purchase!"}, {'customer_name': 'Ray Bradberry', 'rating': 5, 'review': "What to say?  It's great!"}, {'customer_name': 'Piel genuina y excelente producto.', 'rating': 5, 'review': 'Celular nuevo, sellado, con todos cable tipo C, manuales y herramienta para extraer SIM.Celular con excelente cámara, muy rápido y con excelente pantalla.'}, {'customer_name': 'Arlanda miles', 'rating': 5, 'review': 'Great phone and photos'}, {'customer_name': 'RMD', 'rating': 5, 'review': "The camera arrived on time as advertised.  It was in a sealed box.  I bought a case and screen protector before using the phone.  I had no problems transferring everything from my old phone to this.  I was missing a couple dates on my calendar.  I didn't want to pay more through my carrier for this phone and it is comparable to the S24 Ultra they are pushing now at $700 more plus trade in.  I kept my S20 FE and got a great new phone!"}, {'customer_name': 'Ricardo/Blanca', 'rating': 5, 'review': 'Awesome phone picture quality is good and battery lasts long. Overall really pleased with purchase'}, {'customer_name': 'Orakey', 'rating': 5, 'review': "A little bit of discount for a year's old model but major functions are pretty much the same."}, {'customer_name': 'Esmarlin Nuñez', 'rating': 5, 'review': 'Tengo solo unos días con el móvil y les puedo decir que es súper excelente, me encanta y la cámara es lo mejor estoy enamorada de mi cel'}, {'customer_name': 'Austin J.', 'rating': 5, 'review': "Had an issue when I ordered where they didn't have the color that I ordered but luckily they had the color I wanted originally and they sent me that one instead. Came in perfect condition with not a scratch on it! Very happy with my purchase!"}, {'customer_name': 'Tanya Harrell', 'rating': 5, 'review': "I upgraded from a galaxy s10 to this s23 ultra. Still learning how it works. Hate the choices for the default sounds, so that is what I focused on 1st. Haven't found a use for the stylus pen. Not exactly sure what to do with it. The phone itself is a great size and works much faster than my old phone. My voicemail app was the only thing that did not tranfer over and I am having to go back to the old way of dialing into Verizon for my voicemail. That is a bummer, but the phone was everything as advertised."}, {'customer_name': 'Ben F.', 'rating': 5, 'review': "I have amended this since my first review. The seller was very cooperative and got a hold of me and we resolved everything. After doing some of my own savvy due diligence since I am a technologist the phone works perfectly you just have to do some tweaking and really understand technology as I do to make them work properly. I do not suggest changing the APN without first consulting your carrier. However the phones were awesome they were as described and the shipping was quick and I would definitely buy from them again. One out of caution depending on your carrier you may or may not be able to use the international version so you should check first because Samsung will not support these phones in the United States cuz they are not sold on the USA site to any carrier or any person so again you best know what you're doing to make the phone work correctly. This seller is kind enough to send the APN settings if you need them as part of his package so kudos to the seller!!"}, {'customer_name': 'BORIS BOCHOUKOV', 'rating': 5, 'review': 'Perfect working phone'}, {'customer_name': 'raj77', 'rating': 5, 'review': 'Love everything about the phone! Brand new at a discounted price. I recommend this seller.'}, {'customer_name': 'Amazon Customer', 'rating': 5, 'review': 'A bit heavier than the S10+ i replaced but such an improvement! Camera is pgenomenal and battery life is so much better!'}, {'customer_name': 'Ross Inguillo', 'rating': 5, 'review': 'The media could not be loaded.\n                \n\n\n\nLooks exactly as it shown'}, {'customer_name': 'Heidi', 'rating': 5, 'review': "I liked the product.  It didn't come with a wall charger but that's okay!"}, {'customer_name': 'Samuel Navarro', 'rating': 5, 'review': 'El celular es nuevo y funciona a la perfección.  Es un modelo con cámaras excepcionales y buenas funciones. 100% satisfecho'}, {'customer_name': 'tommy', 'rating': 5, 'review': "I feel like no has truly gave a review like this to let everyone know own about the signal after making the purchase I didn't revise I had bought a international which lead me to find out my old was a inter national phone as well ... I can speak for any phone company other than Verizon here but as soon as I plug it in I had 5g in America with a international phone I would say this phone rival the usa version for slightly less cost then buying one at your local Verizon dealer .  What I really liked about this phone is the ability to go into the menu and choice my signal aka the list has a option for 5g,4g,3g,2g even though 2g and 3g are shut off on internet  for American  users. Ultimately  the 4g with this phone has revealed the LG v30 which could get 150 MG download speed where as this phone gets 130 MG but that might just be the difference  in antena slightly"}, {'customer_name': 'Andy tattoo', 'rating': 5, 'review': 'Llego un día antes estoy muy contento 😌'}, {'customer_name': 'Joe W', 'rating': 5, 'review': 'I do everything on my phone, and I need a high powered workhorse that can take the workload. And this phone is it. The battery last easily for two days. Granted, I do not stream a lot of video, or play games much.'}, {'customer_name': 'Jatin Patel', 'rating': 5, 'review': 'This phone is a beast. It runs fast. Has good battery life. Takes great pictures in day time and decent in low light/night time.In hand feel is very premium. The display is gorgeous and the fingerprint sensor works like a charm.Bought an international unlocked device from the Amazon US and works flawlessly.'}, {'customer_name': 'Patricia', 'rating': 5, 'review': 'I was skeptical about buying on Amazon. But this is a legit Samsung s23 ultra. Works with the mobile in the United States. At least in Missouri.'}]
    


```python
print("Total number of reviewes received: ", len(reviews))
```

    Total number of reviewes received:  97
    


```python
# Transfering the webscrapped data into a dataframe
if new_data:
    df = pd.DataFrame.from_dict(reviews)
    df.sort_values(by=["rating"], ascending=False, inplace=True)
    df.reset_index(inplace=True)
    df.to_pickle("amazon_reviews_df")  # storing df to ease its future usage
    df.head(5)
else:
    df = pd.read_pickle("amazon_reviews_df")
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>customer_name</th>
      <th>rating</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>96</td>
      <td>Patricia</td>
      <td>5</td>
      <td>I was skeptical about buying on Amazon. But th...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>81</td>
      <td>Orakey</td>
      <td>5</td>
      <td>A little bit of discount for a year's old mode...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67</td>
      <td>Tumbilah</td>
      <td>5</td>
      <td>Camera quality is amazingCharges wellGood stor...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>68</td>
      <td>Kayannaphora</td>
      <td>5</td>
      <td>I really wanted the S23 Ultra because I like t...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>69</td>
      <td>Regina Daniels</td>
      <td>5</td>
      <td>I wrote a review earlier BUT I did not know a ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>92</th>
      <td>26</td>
      <td>Karina M</td>
      <td>1</td>
      <td>False Advertisement</td>
    </tr>
    <tr>
      <th>93</th>
      <td>27</td>
      <td>Clair Wiederholt</td>
      <td>1</td>
      <td>Amazon is deducting a $132.60 re-stocking fee ...</td>
    </tr>
    <tr>
      <th>94</th>
      <td>28</td>
      <td>Sharon</td>
      <td>1</td>
      <td>My phone is defective. Screen flickers, slow s...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>29</td>
      <td>James Moses</td>
      <td>1</td>
      <td>The ad says unlocked for all carriers. This ph...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0</td>
      <td>The Living Waters Farm</td>
      <td>1</td>
      <td>This phone is glitchy brand new out of the box...</td>
    </tr>
  </tbody>
</table>
<p>97 rows × 4 columns</p>
</div>



### Preparing Review Data for Display 

# extracting API reponse in json file
temp_dict = response.json()

# declaring dataframe to collect final review data
df = pd.DataFrame()

df["review"] = [t["body"] for t in temp_dict["reviews"]]
df["review date"] = [(t["date"]) for t in temp_dict["reviews"]]
df["ratings"] = [t["rating"] for t in temp_dict["reviews"]]
df["review title"] = [t["title"] for t in temp_dict["reviews"]]
df["user name"]= [t["user-name"] for t in temp_dict["reviews"]]
df["verified"] = [t["verified"] for t in temp_dict["reviews"]]

# triming trailiing text from date column 
df["review date"] = df["review date"].str[32:]

# diplaying dataframe data
df

## Review Summary Generation

#### Develop Data Summary


```python
# Developing Summary of Reviews for Each Web
amz_cust_reviews = df["review"]
amz_reviews_str = "".join(each for  each in amz_cust_reviews)
print(f"Total length of all reviews text chain is {len(amz_reviews_str)} characters.")
```

    26721
    


```python
# Storing review data into different formats 
# converting the dataframe to CSV format for checking purpose
if new_data:
    df.to_csv("amz_reviews.csv", mode="w", index=False)                       # storing in CSV format
    file = open('./review_docs/amz_reviews.txt','w', encoding='utf-8')        # storing in text format
    file.writelines(amz_reviews_str)
    file.close()
```


```python
# Setting LLM 
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
print("LLM gets loaded successfully")
```

#### Check for Review Content Length


```python
# Counting AutoScraper output tokens

def count_tokens(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

total_tokens = count_tokens(str(amz_reviews_str), "cl100k_base")
print("Total actual number of input tokens =", total_tokens)
```

    Total actual number of input tokens = 6026
    


```python
if total_tokens <= 3500: 
    print("Reveiew summary will be generated using the small content context method.")
    
    summary_statement = """You are an expeienced copy writer providing a world-class summary of product reviews {reviews} from numerous customers \
                        on a given product from different leading e-commerce platforms. You write summary of all reviews for a target audience \
                        of wide array of product reviewers ranging from a common man to an expeirenced product review professional."""
    summary_prompt = PromptTemplate(input_variables = ["reviews"], template=summary_statement)
    llm_chain = LLMChain(llm=llm, prompt=summary_prompt)
    amz_review_summary_smp = llm_chain.run(amz_reviews_str)
    print("Amazon Review Summary: \n\n", amz_review_summary_smp)
```

##### Define Function for Sentiment Analysis


```python
# define a function for sentiment analysis
# https://python.langchain.com/docs/use_cases/tagging/

class Classification(BaseModel):
    Overall_Sentiment: str = Field(..., enum=["Positive", "Neutral", "Negative"])
    Review_Aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    
tagging_prompt = ChatPromptTemplate.from_template(
    """
    Extract the  properties mentioned in the 'Classification' function from the following text.
    Paragraph:
    {input}
    """
)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125").with_structured_output(
    Classification
)

tagging_chain = tagging_prompt | llm
```

### Generating Customer Review Sentiment for smaller inputs


```python
if total_tokens <= 3500:
    output_smp = tagging_chain.invoke({"input": amz_review_summary_smp})
    print("Customer Reviews' Sentiment \n\n", output_smp)
    print("\n *** PROGRAM EXECUTION ABORTED HERE ***")
    raise StopExecution
```


```python
# Splitting the doc into sizeable chunks

raw_documents = TextLoader("./review_docs/amz_reviews.txt", encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=0, separators=["\n\n", "\n", " ", ""])
split_text = text_splitter.split_documents(raw_documents)
# docs = [Document(page_content=each) for each in split_text]
print("Total number of documents =", len(docs))
```

    Total number of documents = 24
    


```python
print(docs[:2])
```

    [Document(page_content='I was skeptical about buying on Amazon. But this is a legit Samsung s23 ultra. Works with the mobile in the United States. At least in Missouri.A little bit of discount for a year\'s old model but major functions are pretty much the same.Camera quality is amazingCharges wellGood storageDual SIMValue for moneyI really wanted the S23 Ultra because I like the curved edges, which my old phone also had, but no phone carrier or store had them in stock. I was nervous about ordering a phone off Amazon, but my order came fast, in the factory packaging, with no issues and a nice little gift (a charger) from the seller. Plus, the phone is everything I wanted: battery lasts for days, tons of storage, simple to transfer data from my old Galaxy, and charges quickly. All I had to do to set it up was transfer the SIM card. Overall, great buy.I wrote a review earlier BUT I did not know a charger did not come with the cell phone.  Normally a SIM Card is included but it was not.  The small items that should be include are not.  My delivery read left in mailbox; "NO" the box (Too large for the mailbox) and my mail was left on the front porch.  Also, the charge is not included.  Basically, you\'re just'), Document(page_content='getting the phone with no SIM Card and the basic Charger.The cell is a great phone.Upon opening the package, I read the introduction card, which states "This phone is Not compatible with Cricket Wireless." So I returned it. I should have simply installed my Cricket Sim, and it probably would have worked fine like my daughters. Live and learn.I thought i commented regarding this phone already, neway, good phone the first day i powered this up it got a bit warm but since then its been about a month i dont have any complaints at all. Highly recommendNo trae cables necesario para la carga de bateríaAfter my husband screen broke and used our warranty to get a nothing one. Instead of upgrading we love this phone so much we keep with it. We haven\'t had any issues with this phone and would give this one 10 stars.After Having Iphone for almost 10 years I decided it was time for a change and haven\'t look back since. I love the S Pen everything about this phone there is only one down fall this phone doesn\'t run on 5G UC only on 5G. Other than that I love this purchase!What to say?  It\'s great!Celular nuevo, sellado, con todos cable tipo C, manuales y herramienta para extraer SIM.Celular con')]
    

### Apply Map Reduce Method 
#### (Summarize large Document)


```python
# Applying map reduce to summarize large document
# https://python.langchain.com/docs/use_cases/summarization/
print("Map Reduce Process is initiated now") 

map_template = """Based on the following docs {docs}, please provide summary of reviews presented in these documents. 
Review Summary is:"""

map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)
```

The ReduceDocumentsChain handles taking the document mapping results and reducing them into a single output. It wraps a generic CombineDocumentsChain (like StuffDocumentsChain) but adds the ability to collapse documents before passing it to the CombineDocumentsChain if their cumulative size exceeds token_max. In this example, we can actually re-use our chain for combining our docs to also collapse our docs.

So if the cumulative number of tokens in our mapped documents exceeds 4000 tokens, then we’ll recursively pass in the documents in batches of \< 4000 tokens to our StuffDocumentsChain to create batched summaries. And once those batched summaries are cumulatively less than 4000 tokens, we’ll pass them all one last time to the StuffDocumentsChain to create the final summary.


```python
# Reduce
reduce_template = """The following is set of summaries: 
{doc_summaries}
Take these document and return your consolidated summary in a professional manner addressing the key points of the customer reviews. 
Review Summary is:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
```


```python
# Run chain
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="doc_summaries")

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=3500,
)
```

Combining our map and reduce chains into one


```python
# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)
```

#### Generating Map Reduce Summary


```python
amz_review_summary_mr = map_reduce_chain.invoke(docs)
```


```python
print("Amazon Review Summary as per Map Reduced Method: \n\n", amz_review_summary_mr['input_documents'][0])
```

    page_content='I was skeptical about buying on Amazon. But this is a legit Samsung s23 ultra. Works with the mobile in the United States. At least in Missouri.A little bit of discount for a year\'s old model but major functions are pretty much the same.Camera quality is amazingCharges wellGood storageDual SIMValue for moneyI really wanted the S23 Ultra because I like the curved edges, which my old phone also had, but no phone carrier or store had them in stock. I was nervous about ordering a phone off Amazon, but my order came fast, in the factory packaging, with no issues and a nice little gift (a charger) from the seller. Plus, the phone is everything I wanted: battery lasts for days, tons of storage, simple to transfer data from my old Galaxy, and charges quickly. All I had to do to set it up was transfer the SIM card. Overall, great buy.I wrote a review earlier BUT I did not know a charger did not come with the cell phone.  Normally a SIM Card is included but it was not.  The small items that should be include are not.  My delivery read left in mailbox; "NO" the box (Too large for the mailbox) and my mail was left on the front porch.  Also, the charge is not included.  Basically, you\'re just'
    

### Apply Refine Method 
#### (Summarize large Document)


```python
# Checking the Refine Method for comparison
# https://medium.com/@abonia/summarization-with-langchain-b3d83c030889
print("Document Refine Method is initiated now")

prompt_template = """
                  Please provide a summary of the following text.
                  TEXT: {text}
                  SUMMARY:
                  """

question_prompt = PromptTemplate(
    template=prompt_template, input_variables=["text"]
)

refine_prompt_template = """
              Write a concise summary of the following text delimited by triple backquotes.
              Return your response in that covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
              """

refine_prompt = PromptTemplate(
    template=refine_prompt_template, input_variables=["text"])

# Load refine chain
chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=question_prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_text",
   output_key="output_text",
)
amz_review_summary_ref = chain.invoke({"input_text": docs}, return_only_outputs=True)
```

#### Generating Refine Method Summary


```python
print("Amazon Review Summary as per Refine Method: \n\n", amz_review_summary_ref['intermediate_steps'][0])
```

    The text discusses the purchase of a Samsung S23 Ultra from Amazon. The buyer was initially skeptical but found the phone to be legitimate and compatible with mobile carriers in the United States. The phone had good camera quality, charging capabilities, storage, and dual SIM functionality. The buyer was happy with the purchase as it met their expectations and came with a charger as a gift from the seller. However, there were some issues with the delivery and missing items such as a charger and SIM card.
    

### Customer Sentiment from Review Summaries


```python
# Generating customer reviews sentiment based on map reduce method summary
output_mr = tagging_chain.invoke({"input": amz_review_summary_mr['input_documents'][0]})
```


```python
print("Sentiment Output for Map Reduce Summary :", output_mr)
```

    Sentiment Output for Map Reduce Summary : Overall_Sentiment='Positive' Review_Aggressiveness=3
    


```python
# Generating customer reviews sentiment based on refine method summary
output_ref = tagging_chain.invoke({"input": amz_review_summary_ref})
```


```python
print("Sentiment Output for Refined Method Summary :", output_ref)
```

    Sentiment Output for Refined Method Summary : Overall_Sentiment='Neutral' Review_Aggressiveness=3
    

### Generating Comparative Output Summary


```python
print("Generating Review Summary and Sentiment Output Dataframe now")

data_output = pd.DataFrame({"Review Summary":[amz_review_summary_mr["input_documents"][0], amz_review_summary_ref["intermediate_steps"][0]], 
                            "Sentiments":[output_mr, output_ref]},
                           index=(["Map Reduce Method", "Refine Method"]))

pd.set_option("display.colheader_justify","center")
pd.set_option('display.max_colwidth', None)
```

### Exporting Summary


```python
data_output.to_pickle("data_output")
data_output.to_csv("data output.csv", mode="w", index=False)
```

### Displaying Summary


```python
data_output = data_output.style.set_properties(**{'text-align': 'left'})
data_output.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])

print("Final Data Output generation has been complete now")
data_output
```




<style type="text/css">
#T_857a2 th {
  text-align: center;
}
#T_857a2_row0_col0, #T_857a2_row0_col1, #T_857a2_row1_col0, #T_857a2_row1_col1 {
  text-align: left;
}
</style>
<table id="T_857a2">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_857a2_level0_col0" class="col_heading level0 col0" >Review Summary</th>
      <th id="T_857a2_level0_col1" class="col_heading level0 col1" >Sentiments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_857a2_level0_row0" class="row_heading level0 row0" >Map Reduce Method</th>
      <td id="T_857a2_row0_col0" class="data row0 col0" >page_content='I was skeptical about buying on Amazon. But this is a legit Samsung s23 ultra. Works with the mobile in the United States. At least in Missouri.A little bit of discount for a year\'s old model but major functions are pretty much the same.Camera quality is amazingCharges wellGood storageDual SIMValue for moneyI really wanted the S23 Ultra because I like the curved edges, which my old phone also had, but no phone carrier or store had them in stock. I was nervous about ordering a phone off Amazon, but my order came fast, in the factory packaging, with no issues and a nice little gift (a charger) from the seller. Plus, the phone is everything I wanted: battery lasts for days, tons of storage, simple to transfer data from my old Galaxy, and charges quickly. All I had to do to set it up was transfer the SIM card. Overall, great buy.I wrote a review earlier BUT I did not know a charger did not come with the cell phone.  Normally a SIM Card is included but it was not.  The small items that should be include are not.  My delivery read left in mailbox; "NO" the box (Too large for the mailbox) and my mail was left on the front porch.  Also, the charge is not included.  Basically, you\'re just'</td>
      <td id="T_857a2_row0_col1" class="data row0 col1" >Overall_Sentiment='Positive' Review_Aggressiveness=3</td>
    </tr>
    <tr>
      <th id="T_857a2_level0_row1" class="row_heading level0 row1" >Refine Method</th>
      <td id="T_857a2_row1_col0" class="data row1 col0" >The text discusses the purchase of a Samsung S23 Ultra from Amazon. The buyer was initially skeptical but found the phone to be legitimate and compatible with mobile carriers in the United States. The phone had good camera quality, charging capabilities, storage, and dual SIM functionality. The buyer was happy with the purchase as it met their expectations and came with a charger as a gift from the seller. However, there were some issues with the delivery and missing items such as a charger and SIM card.</td>
      <td id="T_857a2_row1_col1" class="data row1 col1" >Overall_Sentiment='Neutral' Review_Aggressiveness=3</td>
    </tr>
  </tbody>
</table>



