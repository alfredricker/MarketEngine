
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime
import time
from selenium import webdriver
import re
import requests


def get_bloomberg_data(symbol,max_page:int=30):
    df = {'Date':[],'Headline':[],'Summary':[]}
    for page in range(max_page):
        url = f"https://www.bloomberg.com/markets2/api/search?page={page}&query={symbol}&sort=time:desc"

        payload = {}
        headers = {
        'authority': 'www.bloomberg.com',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'cookie': '_gcl_au=1.1.1258867486.1687890016; _pxvid=43f2ef4b-1517-11ee-9c98-4d76c2da6132; _rdt_uuid=1687890040318.60c499b1-45fd-4a4e-8f20-2f74cd36be33; professional-cookieConsent=new-relic|perimeterx-bot-detection|perimeterx-pixel|google-tag-manager|google-analytics|microsoft-advertising|eloqua|adwords|linkedin-insights; drift_aid=50f77712-c0e7-4dca-be00-1a3be2328a99; driftt_aid=50f77712-c0e7-4dca-be00-1a3be2328a99; _ga_NNP7N7T2TG=GS1.1.1690421511.1.1.1690421544.27.0.0; optimizelyEndUserId=oeu1691579311263r0.14078048370792384; _sp_v1_data=585912; _sp_su=false; _gid=GA1.2.899983733.1691579312; ccpaUUID=8ddf4e4d-c0f6-4360-a8b8-a8380fef7366; dnsDisplayed=true; ccpaApplies=true; signedLspa=false; bbgconsentstring=req1fun1pad1; _gcl_aw=GCL.1691579312.CjwKCAjw8symBhAqEiwAaTA__DO-uluShYNXs3DHq9qThKK2LjCpEYvmtIQC0s2nkstqwB9aO3_W5xoCYs8QAvD_BwE; _gcl_dc=GCL.1691579312.CjwKCAjw8symBhAqEiwAaTA__DO-uluShYNXs3DHq9qThKK2LjCpEYvmtIQC0s2nkstqwB9aO3_W5xoCYs8QAvD_BwE; bdfpc=004.7462060169.1691579312202; _reg-csrf=s%3AlyBG8tgXEI9gNgMwydkfiRDm.RlyuzWcn50CU7nh9OqEZ95DENiWW7I6ne4qIka7hhS8; pxcts=139f8f51-36a5-11ee-ab90-6964554c7146; _scid=150ea57a-2797-4865-a5c9-ae5f03bd0f1b; _fbp=fb.1.1691579312789.341111564; agent_id=5b10caf2-1327-404d-bdd9-a996d347e790; session_id=020044b7-d1ac-4592-878e-cec443ac6e02; session_key=79b794f0c70c696278e62a0cf85e93415508cbaa; gatehouse_id=9d7077e9-22af-491f-a236-042c63c2a8f0; geo_info=%7B%22countryCode%22%3A%22US%22%2C%22country%22%3A%22US%22%2C%22cityId%22%3A%225368361%22%2C%22provinceId%22%3A%225332921%22%2C%22field_p%22%3A%22E6A909%22%2C%22field_d%22%3A%22rr.com%22%2C%22field_mi%22%3A-1%2C%22field_n%22%3A%22hf%22%2C%22trackingRegion%22%3A%22US%22%2C%22cacheExpiredTime%22%3A1692184112798%2C%22region%22%3A%22US%22%2C%22fieldMI%22%3A-1%2C%22fieldN%22%3A%22hf%22%2C%22fieldD%22%3A%22rr.com%22%2C%22fieldP%22%3A%22E6A909%22%7D%7C1692184112798; _li_dcdm_c=.bloomberg.com; _lc2_fpi=b1166d620485--01h7czqv6gbrbhddx2qjmrg1mb; _gac_UA-11413116-1=1.1691579314.CjwKCAjw8symBhAqEiwAaTA__DO-uluShYNXs3DHq9qThKK2LjCpEYvmtIQC0s2nkstqwB9aO3_W5xoCYs8QAvD_BwE; _sctr=1%7C1691553600000; seen_uk=1; exp_pref=AMER; ln_or=eyI0MDM1OTMiOiJkIn0%3D; _cc_id=cb4452165a13807982d81de51e7402ec; panoramaId=7071f388f239d3ecb953ab1733d316d5393826fd6dfa307fa39b72341051eee0; panoramaIdType=panoIndiv; afUserId=044f5126-bd6b-48c9-b84d-1a6205bfd708-p; AF_SYNC=1691579323016; _sp_v1_p=192; _parsely_session={%22sid%22:4%2C%22surl%22:%22https://www.bloomberg.com/search?query=AAPL&page=3&sort=time:desc%22%2C%22sref%22:%22%22%2C%22sts%22:1691627981370%2C%22slts%22:1691625583270}; _parsely_visitor={%22id%22:%22pid=fb1a7e688db3d4e11dc092a353d07cce%22%2C%22session_count%22:4%2C%22last_session_ts%22:1691627981370}; _sp_v1_ss=1:H4sIAAAAAAAAAItWqo5RKimOUbLKK83J0YlRSkVil4AlqmtrlXQGVlk0kYw8EMOgNhaXkfSQGOiwGnzKYgF_3pWTZQIAAA%3D%3D; _sp_krux=true; euconsent-v2=CPwSZUAPwSZUAAGABCENDPCgAP_AAEPAABpYH9oB9CpGCTFDKGh4AKsAEAQXwBAEAOAAAAABAAAAABgQAIwCAEASAACAAAACGAAAIAIAAAAAEAAAAEAAQAAAAAFAAAAEAAAAIAAAAAAAAAAAAAAAAIEAAAAAAUAAEFAAgEAAABIAQAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgYtAOAAcAD8ARwA7gCBwEHAQgAiIBFgC6gGvAWUAvMBggDFgDwkBcABYAFQAMgAcABAADIAGgARAAjgBMACeAH4AQgAjgBSgDKgHcAd4A9gCTAEpAOIAuQBkgQAGARwAnYCmw0AEBcgYACApsRABAXIKgAgLkGQAQFyDoDQACwAKgAZAA4ACAAFwAMgAaABEACOAEwAJ4AXQAxAB-AFGAKUAZQA7wB7AEmAJSAcQA6gC5AGSDgAoAFwBHAEcAJ2ApshANAAWABkAFwATAAxACOAFKAMqAdwB3gEpAOoAsoBchAACARwlAMAAWABkADgARAAjgBMADEAI4AUYA7wDqAMkJAAgALgEcKQFAAFgAVAAyABwAEAANAAiABHACYAE8AMQAfgBRgClAGUAO8AlIB1AFyAMkKABAALgAyAIOAps.YAAAAAAAAAAA; consentUUID=87aa1011-a199-4670-9bfa-e8a090dc0452_22; country_code=US; _reg-csrf-token=qtPaStnF-joAKNZrqCPnkMRvrKeSRd3V9FQQ; _user-data=%7B%22status%22%3A%22anonymous%22%7D; _last-refresh=2023-8-10%201%3A2; _scid_r=150ea57a-2797-4865-a5c9-ae5f03bd0f1b; __sppvid=43fc3cd7-9fee-4242-ad89-772dd26f0988; _uetsid=13d2691036a511ee9072e745523f4a97; _uetvid=524378e0151711eeb02ba54b6db7a142; _px3=006a2178000e88b5a8383e394d20abe4dec20fe6b83fec951bb5462fd849eb60:FwQbebtd0GCPQQcjQh6Az/v7P3oymeKuiyEhxRdWi+jkaqslxMxMp6XHIFTbdnbVR18c+tpQWFgGzXZeME4ARw==:1000:6mHvOZ7bghHs9Vwo3KeoDBalw4QnYCNkRQ86Wex8kkb/kSAhQS7Ez3WTj+2sp8oIfTypK/JaqIz6uRQ0XEV+riKpp5m6hFvxfiXPlKY2/RUzrecx5lF5mVDetdpb2t+MFvJp7X7XZTaxejFyjHlzxShVPX0Rec9UipbH7A9OdQVnff7EbL7t+6oJc2BXDTPSHOLsISghRszRcWl9R/9eNQ==; _px2=eyJ1IjoiOTY0ZTJkODAtMzcxOS0xMWVlLWJlYTUtYmZhODc3YWIwYmY4IiwidiI6IjQzZjJlZjRiLTE1MTctMTFlZS05Yzk4LTRkNzZjMmRhNjEzMiIsInQiOjE2OTE2Mjk2NTM4NTQsImgiOiIxYWU0NTYwNWNhOWU0OWRlMGM3NmQyZWYyN2Y2ZTlmNjA0NzE3ZWQ2NDdmMGRiMTIwMGY3ZDAyYjFkOWQ2Y2IyIn0=; panoramaId_expiry=1692234153858; __gads=ID=76572d94ef31506c:T=1691579321:RT=1691629354:S=ALNI_MYlt1TkN2cOqMPx-hpXB8MARppYBQ; __gpi=UID=000009b26c3c0a8a:T=1691579321:RT=1691629354:S=ALNI_MbuI-GU0H0AoY52S8l18G7ro2rJQA; _ga=GA1.2.1454142836.1687890016; _pxde=fc60b9dadff19404c07f24b80fdb9b6d6a574085838e971e8f4b42cf3d6b5154:eyJ0aW1lc3RhbXAiOjE2OTE2Mjk0ODc5NjgsImZfa2IiOjAsImlwY19pZCI6W119; _gat_UA-11413116-1=1; _ga_GQ1PBLXZCT=GS1.1.1691627960.4.1.1691629514.56.0.0; exp_pref=AMER; country_code=US',
        'newrelic': 'eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjE5ODI2OTciLCJhcCI6IjE0MDk1Mjc5OCIsImlkIjoiYjI3ODk2YWU0ZDNiNGRiYSIsInRyIjoiNGQ3MjY1MmZiMjRmNDUxNzNjZTIwOTY3ZTFkNjZkZDAiLCJ0aSI6MTY5MTYyOTUxNDUxNywidGsiOiIyNTMwMCJ9fQ==',
        'referer': f'https://www.bloomberg.com/search?query={symbol}&page={page}',
        'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'traceparent': '00-4d72652fb24f45173ce20967e1d66dd0-b27896ae4d3b4dba-01',
        'tracestate': '25300@nr=0-1-1982697-140952798-b27896ae4d3b4dba----1691629514517',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }

        response = requests.request("GET", url, headers=headers, data=payload)
        j = response.json()
        try:
            for result in j["results"]:
                df['Date'].append(result["publishedAt"])
                df['Headline'].append(result["headline"])
                df['Summary'].append(result["summary"])
        except KeyError:
            continue

    df = pd.DataFrame(df)
    data = df.to_json(orient='records',date_format='iso')
    with open(f'web_scraping/{symbol}_bloomberg.dat','w') as file:
        file.write(data)

#from requests_html import HTMLSession


def get_reuters_data(company,max_offset:int=50):
    df = {'Date':[],'Headline':[],'Summary':[]}

    for offset in range(max_offset):
        offset *= 20    
        #this was a pain in the ass to figure out
        url = f"https://www.reuters.com/pf/api/v3/content/fetch/articles-by-search-v2?query=%7B%22keyword%22%3A%22{company}%22%2C%22offset%22%3A{offset}%2C%22orderby%22%3A%22display_date%3Adesc%22%2C%22size%22%3A20%2C%22website%22%3A%22reuters%22%7D&d=151&_website=reuters"

        payload = {}
        headers = {
        'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
        'Referer': f'https://www.reuters.com/site-search/?query={company}&offset={offset}',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'sec-ch-ua-platform': '"Windows"',
        'Cookie': 'reuters-geo={"country":"US", "region":"-"}'
        }

        response = requests.request("GET", url, headers=headers, data=payload)
        j = response.json()

        try:
            for item in j["result"]["articles"]:
                date = item["published_time"]
                title = item["basic_headline"]
                description = item["description"]
                df['Date'].append(date)
                df['Headline'].append(title)
                df['Summary'].append(description)
        except: #in case one page of reuters data isn't formatted the same
            input("Change IP then hit enter to continue: ")
            continue   

    df = pd.DataFrame(df)
    data = df.to_json(orient='records',date_format='iso')
    with open(f'web_scraping/{company}_reuters.dat','w') as file:
        file.write(data)


#looks like I can only get data as far back as 2020... so I'm not going to use marketwatch
def get_marketwatch_data(company,max_page:int=5000):
    data = {'Date':[],'Headline':[]}

    for page in range(max_page):
        url = f"https://www.marketwatch.com/search/moreHeadlines?q={company}&ts=0&partial=true&tab=All%20News&pageNumber={page}"

        payload = {}
        headers = {
        'authority': 'www.marketwatch.com',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'cookie': 'optimizelyEndUserId=oeu1691638315857r0.7087681223582081; ccpaApplies=true; ccpaUUID=f1b52e2b-831e-4661-a601-26f45b9c22e3; ab_uuid=2bb6b0db-e60c-4a58-8c6f-1a7ffe71c69d; AMCVS_CB68E4BA55144CAA0A4C98A5%40AdobeOrg=1; _cls_v=71b4337b-4d2a-4857-8ab9-25e547a46069; _cls_s=cb9ce4cd-7044-42e8-b7e0-997bafce5259:0; _pcid=%7B%22browserId%22%3A%22ll4lrzf3xsz5befm%22%7D; cX_P=ll4lrzf3xsz5befm; usr_bkt=HY0f8Of9M1; _pctx=%7Bu%7DN4IgrgzgpgThIC4B2YA2qA05owMoBcBDfSREQpAeyRCwgEt8oBJAEzIE4AmHgZi4CsvAIwB2DqIAMADkHTRvEAF8gA; s_ecid=MCMID%7C07811295556865454993086478630926961519; AMCV_CB68E4BA55144CAA0A4C98A5%40AdobeOrg=1585540135%7CMCIDTS%7C19580%7CMCMID%7C07811295556865454993086478630926961519%7CMCAAMLH-1692243117%7C9%7CMCAAMB-1692243117%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1691645517s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C4.4.0; s_cc=true; _uetsid=75870c20372e11eeb7e4b57f3021c061; _uetvid=758743f0372e11eeaf7773af3d00e586; _rdt_uuid=1691638317494.509efc6d-b206-4dad-ab21-fa1245ae3898; _gcl_aw=GCL.1691638318.CjwKCAjw8symBhAqEiwAaTA__Ory9tIPdbQkDnXN5UtbxHrw9yWDTdXcRgrZFKSNTkdxN0A8UQViuBoCVrgQAvD_BwE; _gcl_au=1.1.1194856486.1691638318; _parsely_session={%22sid%22:1%2C%22surl%22:%22https://store.marketwatch.com/shop/us/us/mwus24ns/?swg=true&trackingCode=aaqzhsyx&cid=MW_SCH_GOO_ACQ_NA&n2IKsaD9=n2IKsaD9&gad=1&gclid=CjwKCAjw8symBhAqEiwAaTA__Ory9tIPdbQkDnXN5UtbxHrw9yWDTdXcRgrZFKSNTkdxN0A8UQViuBoCVrgQAvD_BwE%22%2C%22sref%22:%22https://www.google.com/%22%2C%22sts%22:1691638317590%2C%22slts%22:0}; _parsely_visitor={%22id%22:%22pid=bf2e4b8d-4749-421a-840f-2d21ae71034b%22%2C%22session_count%22:1%2C%22last_session_ts%22:1691638317590}; cX_G=cx%3Aaay94xhyxbv11gicm2zvcbs06%3A118c0xph4rjn9; permutive-id=6c6c2bca-c05c-4138-88d6-6e1f360c4fc5; mw_loc=%7B%22Region%22%3A%22CA%22%2C%22Country%22%3A%22US%22%2C%22Continent%22%3A%22NA%22%2C%22ApplicablePrivacy%22%3A0%7D; gdprApplies=false; fullcss-home=site-60d04d1451.min.css; icons-loaded=true; pushly.user_puuid=KjxevIlduSipjWShxOsKVchz9Kkg0XwA; _lr_geo_location_state=CA; _lr_geo_location=US; sso_fired_at=1691638332775; letsGetMikey=enabled; dnsDisplayed=false; signedLspa=false; _pbjs_userid_consent_data=6683316680106290; _pubcid=264ea218-4b67-4419-bbe9-75518d56c802; _ncg_domain_id_=1a8cf395-3657-4492-be2f-7e6cb4e543a6.1.1691638334184.1754710334184; _ncg_sp_ses.f57d=*; _ncg_id_=be3e9db5-dba4-4a0e-8345-070478527d6b; ln_or=eyIzOTQyNDE3IjoiZCJ9; _fbp=fb.1.1691638335144.1497534032; _dj_sp_id=e5a66968-eecf-41a1-a2f3-f6bb63e46721; _ncg_g_id_=469a5be9-2474-419c-b308-0c37a76461fc.3.1691638334.1754710334184; _pcus=eyJ1c2VyU2VnbWVudHMiOm51bGx9; s_sq=djglobal%252Cdjwsj%3D%2526pid%253Dhttps%25253A%25252F%25252Fwww.marketwatch.com%25252F%2526oid%253D%25250A%252520%252520%252520%252520%252520%252520%2526oidt%253D3%2526ot%253DSUBMIT; fullcss-quote=quote-ccd11d2396.min.css; recentqsmkii=Stock-US-AAPL; _lr_retry_request=true; _lr_env_src_ats=false; _lr_sampling_rate=0; _pnss=dismissed; fullcss-section=section-15f53c310c.min.css; _dj_ses.cff7=*; _dj_id.cff7=.1691638335.2.1691638415.1691638346.1cb581d7-8c38-46ee-a985-6585241dc745; _ncg_sp_id.f57d=be3e9db5-dba4-4a0e-8345-070478527d6b.1691638334.1.1691638416..355bcfa9-1f16-4ad3-af68-c3a5932ac81c..11fff203-d6ab-4d9b-a1a2-a78d6c9faa60.1691638334563.12; cto_bundle=tfCVBF94MEJVYldSQ0NVaHFpalpDbTJFN1pEMVlDS0hoOENBVEtoQk03RTNTWFBXTlVlc0lIVXhPYlNjV1pUUVNtU0s4RkZPaHNBMlFtc0h0VzJtbXdWcXF6NVFuSXRhYUliVHFmUnBJcGxjZ0xaaUZHRFVIczRaMXozN0RhQmxsNkZkTUhZdkVPZEJQUnJNemVBbFRFdlgxb2clM0QlM0Q; utag_main=v_id:0189dd803edd000cc3ddf5e2d12b0506f00e106700aee$_sn:1$_ss:0$_st:1691640422068$ses_id:1691638316765%3Bexp-session$_pn:4%3Bexp-session$vapi_domain:marketwatch.com$_prevpage:undefined%3Bexp-1691642222070; s_tp=4549; s_ppv=MW_Search%2C77%2C77%2C3520; __gads=ID=5f76ae834c3d2c87:T=1691638332:RT=1691638686:S=ALNI_MYeQa6qXb5p7Jn8m05Wegyo5l6AwQ; __gpi=UID=000009b28943ba2b:T=1691638332:RT=1691638686:S=ALNI_Ma3mi2vCUEEu9MQ552iyJ8f_OnyLg; gdprApplies=false',
        'newrelic': 'eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjE2ODQyNzMiLCJhcCI6Ijc1NDg5NDA2MCIsImlkIjoiNDJkMGIwY2I5MzkxNDJjOSIsInRyIjoiMmMyNjQxMzhhMDAxZWIyODhjMDIxODM5MTY5OTQxMDAiLCJ0aSI6MTY5MTYzODY5NjE5MCwidGsiOiIxMDIyNjgxIn19',
        'referer': f'https://www.marketwatch.com/search?q={company}&ts=0&tab=All%20News&pageNumber={page}',
        'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'traceparent': '00-2c264138a001eb288c02183916994100-42d0b0cb939142c9-01',
        'tracestate': '1022681@nr=0-1-1684273-754894060-42d0b0cb939142c9----1691638696190',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }

        response = requests.request("GET", url, headers=headers, data=payload)
        response_text = response.text
        soup = BeautifulSoup(response_text, 'html.parser')

        # Find all elements with the class "article__headline"
        headline_elements = soup.find_all(class_="article__headline")

        # Iterate through headline elements and extract headlines and dates
        for headline_element in headline_elements:
            headline_text = headline_element.a.get_text(strip=True)
            
            # Find the corresponding timestamp element
            timestamp_element = headline_element.find_next(class_="article__timestamp")
            date_text = timestamp_element.get_text(strip=True)
            
            if headline_text and date_text:
                data['Date'].append(date_text)
                size = len(data['Date'])
                if size%1000==0:
                    print(size)
                data['Headline'].append(headline_text)
        
    df = pd.DataFrame(data)
    j = df.to_json(orient='records',date_format='iso')
    with open(f'web_scraping/{company}_marketwatch.dat','w') as file:
        file.write(j)    


#please don't change your website format barrons
#pain. I get blocked before I can even get data before 2022... I'll have to use a try except
#even more pain... the dates only go back to 2022 for Barron's. I'll have to limit my usage to new stocks
def get_barrons_data(company,max_page:int=60):
    
    df = {'Date':[],'Headline':[],'Summary':[]}
    for page in range(max_page):
        url = f'https://www.barrons.com/search?id=%7B%22query%22%3A%7B%22not%22%3A%5B%7B%22terms%22%3A%7B%22key%22%3A%22SectionType%22%2C%22value%22%3A%5B%22NewsPlus%22%5D%7D%7D%5D%2C%22and%22%3A%5B%7B%22terms%22%3A%7B%22key%22%3A%22languageCode%22%2C%22value%22%3A%5B%22en%22%2C%22en-us%22%5D%7D%7D%2C%7B%22date%22%3A%7B%22key%22%3A%22liveDate%22%2C%22value%22%3A%222023-08-10T19%3A59%3A59-04%3A00%22%2C%22operand%22%3A%22LessEquals%22%7D%7D%2C%7B%22date%22%3A%7B%22key%22%3A%22liveDate%22%2C%22value%22%3A%222022-08-10T00%3A00%3A00%2B00%3A00%22%2C%22operand%22%3A%22GreaterEquals%22%7D%7D%2C%7B%22terms%22%3A%7B%22key%22%3A%22Product%22%2C%22value%22%3A%5B%22Barrons.com%22%2C%22Barrons.com%20Automated%20Market%20Wraps%22%2C%22Barrons%20Blogs%22%2C%22Barrons%20Advisor%20Credits%20Video%22%2C%22Barrons%20Broadband%20Video%22%2C%22Barrons%20Summit%20Video%22%2C%22Barrons%20Video%20Live%20Q%26A%22%2C%22Barrons.com%20Webstory%22%2C%22Barrons%20Live%20Coverage%22%5D%7D%7D%5D%2C%22or%22%3A%5B%7B%22query_string%22%3A%7B%22value%22%3A%22apple%22%2C%22default_or_operator%22%3Atrue%2C%22parameters%22%3A%5B%7B%22property%22%3A%22headline%22%2C%22boost%22%3A3%7D%2C%7B%22property%22%3A%22keywords%22%2C%22boost%22%3A4%7D%2C%7B%22property%22%3A%22byline%22%2C%22boost%22%3A3%7D%2C%7B%22property%22%3A%22body%22%2C%22boost%22%3A4%7D%2C%7B%22property%22%3A%22section_name%22%2C%22boost%22%3A5%7D%5D%7D%7D%2C%7B%22full_text%22%3A%7B%22value%22%3A%22{company}%22%2C%22match_phrase%22%3Atrue%2C%22parameters%22%3A%5B%7B%22property%22%3A%22headline%22%2C%22boost%22%3A3%7D%2C%7B%22property%22%3A%22body%22%2C%22boost%22%3A4%7D%5D%7D%7D%5D%7D%2C%22sort%22%3A%5B%7B%22key%22%3A%22relevance%22%2C%22order%22%3A%22desc%22%7D%5D%2C%22count%22%3A20%7D%2Fpage%3D{page}&type=allesseh_search_full_v2'

        payload = {}
        headers = {
        'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
        'tracestate': '1022681@nr=0-1-1684273-1385912469-255f303fef1948f1----1691666216001',
        'traceparent': '00-6964ae46184ea5919bfb3c1175932e00-255f303fef1948f1-01',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'newrelic': 'eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjE2ODQyNzMiLCJhcCI6IjEzODU5MTI0NjkiLCJpZCI6IjI1NWYzMDNmZWYxOTQ4ZjEiLCJ0ciI6IjY5NjRhZTQ2MTg0ZWE1OTE5YmZiM2MxMTc1OTMyZTAwIiwidGkiOjE2OTE2NjYyMTYwMDEsInRrIjoiMTAyMjY4MSJ9fQ==',
        'Referer': f'https://www.barrons.com/search?mod=DNH_S&query={company}&page={page+1}',
        'sec-ch-ua-platform': '"Windows"'
        }
        try:
            response = requests.request("GET", url, headers=headers, data=payload)
            data = response.json()
        except:
            input("Change IP address and press enter to continue: ")
            response = requests.request("GET", url, headers=headers, data=payload)
            data = response.json()
        ids = [item["id"] for item in data["collection"]]
        for id in ids:
            #print(id)
            if id.startswith("lc"):
                continue
            url = f"https://www.barrons.com/search?id={id}&type=article%7Ccapi"

            id_payload = {}
            id_headers = {
            'authority': 'www.barrons.com',
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'cookie': 'wsjregion=na%2Cus; gdprApplies=false; ccpaApplies=true; ab_uuid=57746ccb-f0e2-4456-acf3-5bf962e1ecd9; usr_bkt=1p6sdUNJWD; _pcid=%7B%22browserId%22%3A%22ll52818318ms5m7h%22%7D; cX_P=ll52818318ms5m7h; _pctx=%7Bu%7DN4IgrgzgpgThIC4B2YA2qA05owMoBcBDfSREQpAeyRCwgEt8oBJAEzIE4AmHgZi4CsvAIwB2DqIAMADkHTRAkAF8gA; _lr_geo_location_state=CA; _lr_geo_location=US; _pbjs_userid_consent_data=6683316680106290; _pubcid=a9c66174-d226-4d66-96b3-b1de5cdb1c63; cX_G=cx%3Aaay94xhyxbv11gicm2zvcbs06%3A118c0xph4rjn9; ccpaUUID=c2d2144e-67e6-4ffb-a000-5fce7bc58548; AMCVS_CB68E4BA55144CAA0A4C98A5%40AdobeOrg=1; AMCV_CB68E4BA55144CAA0A4C98A5%40AdobeOrg=1585540135%7CMCIDTS%7C19580%7CMCMID%7C07811295556865454993086478630926961519%7CMCAAMLH-1692270742%7C9%7CMCAAMB-1692270742%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1691673142s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C4.4.0; s_cc=true; _rdt_uuid=1691665942919.a6e1c4be-6736-4b59-bb15-f8bde9c1518d; _fbp=fb.1.1691665943068.1240242682; _cls_v=451db5d3-3a0a-4094-b5a0-40ddf833857a; _cls_s=f3b88220-d350-41e8-b5b9-c6a0cfbb0d99:0; _ncg_domain_id_=0ec0072f-4db3-49d7-ba37-b2560556712d.1.1691665942904.1754737942904; _parsely_session={%22sid%22:1%2C%22surl%22:%22https://www.barrons.com/%22%2C%22sref%22:%22https://www.google.com/%22%2C%22sts%22:1691665943811%2C%22slts%22:0}; _parsely_visitor={%22id%22:%22pid=5e443952-8844-4822-9928-a578fd3058a9%22%2C%22session_count%22:1%2C%22last_session_ts%22:1691665943811}; _ncg_g_id_=4d40a8c1-d098-4817-9b73-adeb1ac41c21.3.1691665459.1754737942904; __gads=ID=c0815736f6fedddf:T=1691665942:RT=1691665942:S=ALNI_MboodYppO-KeovmgXjnvWNX5fiBHw; __gpi=UID=000009b291b15d47:T=1691665942:RT=1691665942:S=ALNI_MaJtq0rnOlFeh7f7piV5qu5GnNPAA; _lr_retry_request=true; _lr_env_src_ats=false; _dj_sp_id=efedd005-ecfa-4efd-b74f-0f17486efade; s_sq=djglobal%3D%2526c.%2526a.%2526activitymap.%2526page%253DBOL_Home_US%252520Home%252520Page%252520Weekday%2526link%253DSearch%252520News%252520%252526%252520Quotes%2526region%253Droot%2526pageIDType%253D1%2526.activitymap%2526.a%2526.c; usr_prof_v2=eyJpYyI6NH0%3D; permutive-id=6c6c2bca-c05c-4138-88d6-6e1f360c4fc5; DJSESSION=country%3Dus%7C%7Ccontinent%3Dna%7C%7Cregion%3Dca; vcdpaApplies=false; regulationApplies=gdpr%3Afalse%2Ccpra%3Atrue%2Cvcdpa%3Afalse; optimizelyEndUserId=oeu1691666193209r0.5358536663461277; ResponsiveConditional_initialBreakpoint=lg; dnsDisplayed=false; signedLspa=false; spotim_visitId={%22visitId%22:%221904069a-ba0a-4552-b5b5-066e23e5e608%22%2C%22creationDate%22:%22Thu%20Aug%2010%202023%2007:16:36%20GMT-0400%20(Eastern%20Daylight%20Time)%22%2C%22duration%22:1}; cto_bundle=MQjLXl9EQURaY0RhRXRScURXR3N2QmQ3V1ltJTJGZzMlMkJrajcyajNPT2Fnak5uZVdXUUJDbFdTMDVCTVJDWW9DaHl4d2ZibjBZVmpJeHlIdW1tMUQ3U3FaekVIQUFPM0c0SFhsZHRPSjZLZ1ZVS0lmYWgwTTZvSSUyRlFVTDJTV1BCVWFaSTNTR0xkb0VhdTZvU0RhalBnJTJCWjBRM3FFQSUzRCUzRA; utag_main=v_id:0189df25c56a0089c6625ae32bf80506f001806700aee$_sn:1$_ss:0$_st:1691669030668$ses_id:1691665941867%3Bexp-session$_pn:10%3Bexp-session$_prevpage:BOL_ResearchTools_Barrons%20Search%3Bexp-1691670830674$vapi_domain:barrons.com; _dj_id.c19f=.1691665943.4.1691667231.1691666841.e009148b-6626-46e1-aca9-a0100440bd68..7d3f95ce-0d69-435d-ac55-ae276a1f1587.1691667231115.1; _ncg_sp_id.e48a=0ec0072f-4db3-49d7-ba37-b2560556712d.1691665943.4.1691667231.1691666841.d4208a56-bd96-43c4-a763-735896527758; _ncg_id_=0ec0072f-4db3-49d7-ba37-b2560556712d; _gcl_au=1.1.690478435.1691667231; s_tp=4377; s_ppv=BOL_ResearchTools_Barrons%2520Search%2C93%2C22%2C4081; sso_fired_at=1691667268066',
            'if-none-match': 'W/"3b20-ajufIk8tn/Y9ejFdBGj9pEolf9I"',
            'newrelic': 'eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjE2ODQyNzMiLCJhcCI6IjEzODU5MTI0NjkiLCJpZCI6ImEyODQ0ZTlhNjMxNGEzMGUiLCJ0ciI6IjI4YWJlNjg4OGZkYjg1YmRiMzczMDE2NTU4Nzc4ZTAwIiwidGkiOjE2OTE2NjcyNjg0NTUsInRrIjoiMTAyMjY4MSJ9fQ==',
            'referer': f'https://www.barrons.com/search?mod=DNH_S&query={company}&page={page+1}',
            'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'traceparent': '00-28abe6888fdb85bdb373016558778e00-a2844e9a6314a30e-01',
            'tracestate': '1022681@nr=0-1-1684273-1385912469-a2844e9a6314a30e----1691667268455',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
            }

            try:
                id_response = requests.request("GET", url, headers=id_headers, data=id_payload)
                id_data = id_response.json()
            except:
                input("Change IP address and press enter to continue: ")
                id_response = requests.request("GET", url, headers=id_headers, data=id_payload)
                id_data = id_response.json()

            time_stamp = id_data["data"]["timestamp"]
            headline = id_data["data"]["headline"]
            #unfortunately, sometimes the articles don't have a summary
            try:
                summary = id_data["data"]["summary"]
            except KeyError:
                summary = ''
            timestamp_milliseconds = int(time_stamp)
            timestamp_seconds = timestamp_milliseconds / 1000

            # Convert timestamp to datetime object
            timestamp_date = datetime.fromtimestamp(timestamp_seconds)
            print(timestamp_date)
            df['Date'].append(timestamp_date)
            df['Headline'].append(headline)
            df['Summary'].append(summary)

    df = pd.DataFrame(df)
    j = df.to_json(orient='records',date_format='iso')
    with open(f'web_scraping/{company}_barrons.dat','w') as file:
        file.write(j)


def get_cnbc_data(company, scroll: int = 30):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run Chrome in headless mode (no UI)
    driver = webdriver.Chrome(options=options)
    
    url = f'https://www.cnbc.com/search/?query={company}&qsearchterm={company}'
    driver.get(url)

    df = {'Date': [], 'Headline': [], 'Summary': []}
    
    # Scroll down the page to load more content
    for _ in range(scroll):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    search_results = soup.find_all(class_='SearchResult-searchResult')
    
    for result in search_results:
        title_element = result.find(class_='Card-title')
        
        if title_element:
            title_text = title_element.get_text()
            
            author_element = result.find(class_='SearchResult-author')
            date_element = result.find(class_='SearchResult-publishedDate')
            preview_element = result.find(class_='SearchResult-searchResultPreview')
            
            author = author_element.get_text() if author_element else "N/A"
            date = date_element.get_text() if date_element else "N/A"
            preview = preview_element.get_text() if preview_element else "N/A"

            df['Date'].append(date)
            df['Headline'].append(title_text)
            df['Summary'].append(preview)
    
    df = pd.DataFrame(df)
    j = df.to_json(orient='records',date_format='iso')
    with open(f'web_scraping/{company}_cnbc.dat','w') as file:
        file.write(j)

    driver.quit()


#HUGGING FACE PRE TRAINED SENTIMENT MODEL
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def roberta_sentiment(sentence:str):
    encoded_text = tokenizer(sentence,return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy() #returns sentiment scores as a np array of the form [neg,neutral,pos]
    scores = softmax(scores)
    #lets return one number in the range (-1,1), -1 being as negative as possible and 1 being as positive as possible
    sentiment = scores[0]*(-1.) + scores[1]*0. + scores[2]*1.
    return sentiment

