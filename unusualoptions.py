from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import os
import time
import pickle
import pandas as pd

'''
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
options.add_argument("--disable-proxy-certificate-handler")
options.add_argument("--disable-content-security-policy")
driver = webdriver.Chrome(options=options)
driver.get("https://www.barchart.com/login")

time.sleep(5)


email = driver.find_element(By.CLASS_NAME,'form-field-login')
password = driver.find_element(By.CLASS_NAME,'form-field-password')
email.send_keys("")
password.send_keys("")
pickle.dump(driver.get_cookies(), open("web_scraping/barchart_cookies.pkl", "wb"))
'''

'''
cookies = {
        "_ccm_inf":"1",
        "_gcl_au":"1.1.239753414.1694226957",
        "usprivacy":"1YNY",
        "_admrla":"2.2-3bd8bb55f72c3979-e4e3ce55-4eb6-11ee-96e3-65bbce279b88",
        "alo_uid":"88940fbf-d7fb-4297-9569-d05811571d1b",
        "_gd_visitor":"5aeb8186-1c10-4c03-8901-08e7db64d0bb",
        "_gd_svisitor":"4d4036175f190000f4cb6663cf01000093725002", 
        "__browsiUID":"e5c0743e-a1cc-4dc4-83eb-3c2999ec742c", 
        "_ga_4HQ9CY2XKK":"GS1.1.1694235995.1.1.1694236036.19.0.0",
        "_gid":"GA1.2.929628719.1695006544",
        "_ga_W0HSBQPE0S":"GS1.1.1695006545.1.0.1695006546.0.0.0",
        "cto_bundle":"adCVpl8zSkdjYmdweGV1UGtWQ0FyZWFlb3VFWGk0MXA0QjZ0ODBQYW9YJTJCZ3UyQUQlMkIlMkJyUDkzbmw4c1Y1T0lMMmt6UTljaTVUWnZYak8wWUElMkJrWWFJdnpsdzQxJTJGMktGYXhualJTQURyN0d4NWUlMkJIUllRODF4NHA4MnM2YUE0ZzQlMkZER1B6ZGNPejJBM1d1V0t3RndQMG1VNmcxYXlFWTNkeE9mamFXNVRKa0UyRjc1d1AlMkJrTWF1NlJSY1RyRDZvSDdjMmgyRVlPcUNtcEhHNUZWNzdxSWwlMkZseTd3JTNEJTNE",
        "webinar158WebinarClosed":"true", 
        "_awl":"2.1695006761.5-51e83fc2fc729d987337916c0e78ce70-6763652d75732d7765737431-0",
        "_ga_PXEQ7G89EH":"GS1.2.1695006548.2.0.1695006765.0.0.0",
        "_hjSessionUser_2563157":"eyJpZCI6ImJlZGE5NDc5LWU1MGEtNWRmYi05MjU0LWRjOTBlZDYwMTQ1NyIsImNyZWF0ZWQiOjE2OTQyMzQzNjkyMjksImV4aXN0aW5nIjp0cnVlfQ==",
        "__gads":"ID=143414b2c8014ab6-221365c9c6e3003e:T=1694226955:RT=1695007224:S=ALNI_MZZ8agNptiDLliyUrTUDfzcJr4HSA", 
        "__gpi":"UID=00000d9129b77101:T=1694226955:RT=1695007224:S=ALNI_Mbae-5YnVwTICOVbgcixHPkk_3hFg",
        "XSRF-TOKEN":"eyJpdiI6IkExY3REcFpxQTdnckRyWC95Q1J5ekE9PSIsInZhbHVlIjoiaVlpL09DMVpaQjQ3UXduSVQvZ0Q2dVZ1Yjg4Y0RVNHhmOEVRMTQvUFJIWXBsbmJhdHVScmtZelYrU3Y1cWNJRW5vVnQ3SjZXaFN0WWkvemZzdXRjTityZTYvOUIyZlNsNURFcVo5TVpwR2srcFVrTFdudzJseERGbTV5RlBhc3EiLCJtYWMiOiI4ODA2ODE1MzVmMmU5YmVjOWVjZDVmNjkwYzZiZjBkMDk1OTY3MjQxMDg1ZjIyMTkwZmI3MzlhMzVjMDdkMzRlIn0%3D",
        "bcFreeUserPageView":"0", 
        "_ga":"GA1.2.1577042263.1694226957",
        "unusualFilter":"%7B%7D", 
        "unusualFilterParams":"%7B%22expirationType%22%3A%5B%22monthly%22%2C%22weekly%22%5D%2C%22symbolType%22%3A%5B%22Call%22%2C%22Put%22%5D%7D",
        "_ga_PE0FK9V6VN":"GS1.1.1695101789.7.1.1695101924.60.0.0"
    }

cwd = os.getcwd()
download_path = os.path.join(cwd, 'data_misc/optionsflow')

options = webdriver.ChromeOptions()
#options.add_argument('--headless')
#options.add_experimental_option('prefs', {
#    'download.default_directory': download_path
#})
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
options.add_argument("--disable-proxy-certificate-handler")
options.add_argument("--disable-content-security-policy")
driver = webdriver.Chrome(options=options)

# Navigate to the web page with the download button
driver.get("https://www.barchart.com/options/unusual-activity/stocks")

with open("web_scraping/barchart_cookies.pkl", "rb") as cookie_file:
    cookies = pickle.load(cookie_file)

# Add the cookies to the WebDriver session
for cookie in cookies:
    driver.add_cookie(cookie)


# Refresh the page to apply the cookies
#driver.get("https://www.barchart.com/options/unusual-activity/stocks?type=historical&historicalDate=2023-05-15&useFilter=1")
#driver.refresh()

try:
    # Use an explicit wait to wait for the element to be present
    wait = WebDriverWait(driver, 10)
    button = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".toolbar-button.download")))

    # Click the button
    button.click()

    # After clicking the button, you can handle any download logic or further actions as needed

except Exception as e:
    print(f"An error occurred: {str(e)}")

# Close the browser window
driver.quit()
'''
from datetime import datetime,timedelta
import functions as fn

file_path = "path/to/your/file.ext"

if os.path.exists(file_path):
    print("File exists.")
else:
    print("File does not exist.")


def format_optionsflow(start_date:datetime,end_date:datetime,print_option:bool=True):
    current_date = start_date
    current_str = datetime.strftime(current_date,'%Y-%m-%d')

    df_dict = {}

    while current_date <= end_date:
        if print_option==True:
            print(current_date)

        try:
            options_df = pd.read_csv(f'data_misc/optionsflow/Unusual-Stock-Options-Activity-{current_str}.csv')
        except Exception as e:
            current_date = current_date + timedelta(days=1)
            current_str = datetime.strftime(current_date,'%Y-%m-%d')
            continue

        for index,value in options_df.iterrows():
            try:
                volOI = float(value['Vol/OI'])
            except:
                volOI = float(value['Vol/OI'].replace(',',''))
            try:
                iv = float(value['IV'].strip('%'))/100
            except:
                iv = str(value['IV']).strip('%')
                iv = iv.replace(',','')
                iv = float(iv)

            delta = float(value['Delta'])
            symbol = value['Symbol']

            if value['Type'] == 'Put':
                volOI*=-1

            if symbol in df_dict:
                #print(df_dict)
                df_dict[symbol]['Date'].append(current_str)
                df_dict[symbol]['volOI'].append(volOI)
                df_dict[symbol]['IV'].append(iv)
                df_dict[symbol]['delta'].append(delta)
            else:
                df_dict[symbol] = {'Date':[current_str],'volOI':[volOI],'IV':[iv],'delta':[delta]}

        current_date = current_date + timedelta(days=1)
        current_str = datetime.strftime(current_date,'%Y-%m-%d')

    for symbol,df in df_dict.items():
        if print_option==True:
            print(symbol)

        if start_date<datetime(2020,5,1):
            df['Date'].insert(0,datetime.strftime(start_date,'%Y-%m-%d'))
            df['volOI'].insert(0,0)
            df['IV'].insert(0,0)
            df['delta'].insert(0,0)

        df = pd.DataFrame(df)

        df_filled = fn.data_fill(df,start_date=start_date,end_date=end_date,damping_columns='all',damping_constant=0.4)
        data = df_filled.to_json(orient='records',date_format='iso')

        with open(f'data_misc/optionsflow/{symbol}.dat','w') as f:
            f.write(data)

#format_optionsflow(datetime(2016,1,1),datetime(2023,6,1))


def get_options_flow(single_date=None,start_date=None,end_date=None, file_method='a',print_option=False):
    if single_date is None and start_date is None and end_date is None:
        print('get options flow error. incorrect combination of inputs')
    
    if single_date is not None:
        current_date=single_date
        end_date=current_date
    else:
        current_date = start_date
    
    current_str = datetime.strftime(current_date,'%Y-%m-%d')

    df_dict = {}

    while current_date <= end_date:
        if print_option==True:
            print(current_date)

        try:
            options_df = pd.read_csv(f'data_misc/optionsflow/Unusual-Stock-Options-Activity-{current_str}.csv')
        except Exception as e:
            current_date = current_date + timedelta(days=1)
            current_str = datetime.strftime(current_date,'%Y-%m-%d')
            continue

        for index,value in options_df.iterrows():
            try:
                volOI = float(value['Vol/OI'])
            except:
                volOI = float(value['Vol/OI'].replace(',',''))
            try:
                iv = float(value['IV'].strip('%'))/100
            except:
                iv = str(value['IV']).strip('%')
                iv = iv.replace(',','')
                iv = float(iv)

            delta = float(value['Delta'])
            symbol = value['Symbol']

            if value['Type'] == 'Put':
                volOI*=-1

            if symbol in df_dict:
                #print(df_dict)
                df_dict[symbol]['Date'].append(current_str)
                df_dict[symbol]['volOI'].append(volOI)
                df_dict[symbol]['IV'].append(iv)
                df_dict[symbol]['delta'].append(delta)
            else:
                df_dict[symbol] = {'Date':[current_str],'volOI':[volOI],'IV':[iv],'delta':[delta]}

        current_date = current_date + timedelta(days=1)
        current_str = datetime.strftime(current_date,'%Y-%m-%d')

    for symbol,df in df_dict.items():
        if print_option==True:
            print(symbol)

        df = pd.DataFrame(df)
        if single_date is not None:
            df_filled = df
        else:
            df_filled = fn.data_fill(df,start_date=start_date,end_date=end_date,damping_columns='all',damping_constant=0.4)

        if file_method=='w':
            data = df_filled.to_json(orient='records',date_format='iso')
            with open(f'data_misc/optionsflow/{symbol}.dat','w') as f:
                f.write(data)

        elif file_method=='a':
            try:
                with open(f'data_misc/optionsflow/{symbol}.dat','r') as f:
                    d = f.read()
                df = pd.read_json(d,convert_axes=True)
                data = pd.concat([df,df_filled],ignore_index=True)
                j = data.to_json(orient='records',date_format='iso')
            except:
                j = df_filled.to_json(orient='records',date_format='iso')

            with open(f'data_misc/optionsflow/{symbol}.dat','w') as f:
                f.write(j)

        else:
            print('get_options_flow error: invalid file_method')
            return 0
            
get_options_flow(start_date=datetime(2023,6,2),end_date=datetime(2023,9,27),print_option=True)


