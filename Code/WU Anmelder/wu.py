from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Starte einen Browser
browser = webdriver.Chrome("/Users/alexanderhillisch/Desktop/Chrome/chromedriver")


# Öffne die Website
browser.get('https://lpis.wu.ac.at/lpis')

# Finde die Matrikelnummer- und Passwort-Felder
time.sleep(1)
matr_nr = browser.find_element(By.XPATH, '//*[@id="login"]/table/tbody/tr[1]/td[2]/input')
password = browser.find_element(By.XPATH, '//*[@id="login"]/table/tbody/tr[2]/td[2]/input')


# Gebe die Anmeldeinformationen ein
matr_nr.send_keys('h12141257')
password.send_keys('@Judo1234AsWa0703')

time.sleep(1)


# Klicke auf die Einloggen-Schaltfläche
login_button = browser.find_element(By.XPATH, '//input[@value="Login"]')
login_button.click()
time.sleep(1)


filter = browser.find_element(By.XPATH, '//*[@id="ea_stupl"]/select')
filter.click()
time.sleep(1)

winf = browser.find_element(By.XPATH, '//*[@id="ea_stupl"]/select/option[6]')
winf.click()


anzeigen = browser.find_element(By.XPATH, '//*[@id="ea_stupl"]/input[4]')
anzeigen.click()
time.sleep(1)




# Hier beginnt das wichtige im Script
def reload_page():
    browser.refresh()

# Definieren Sie die Uhrzeit, zu der die Seite neu geladen werden soll
reload_time = "14:00:00"


#Scrollen--> Eventuell nicht nötig
#browser.execute_script("window.scrollTo(0, 1000);")





# Jetzt Wird die LV Ausgewählt 
# -------------------> Hier ändern <-------------------

lv = browser.find_element(By.XPATH, "/html/body/table[2]/tbody/tr[8]/td[1]/span[3]/a")
lv.click()
time.sleep(1)


# Hier wird der Anmeldebutton geklickt
# -------------------> Hier ändern <-------------------


def click_anmeldebutton():
    lv_id = browser.find_element(By.XPATH, '//*[@id="SPAN_483265_247225"]/input[10]')
    print("LV ID gefunden")
    lv_id.click()
    print("LV ID gefunden und auf anmelden geklickt!")
    time.sleep(1)
    return
    

try:
    while True:
        current_time = time.strftime("%H:%M:%S")
        
        if browser.find_element(By.XPATH, '/html/body/table[2]/tbody/tr[1]/td[4]/div').text == "ABmelden":
            print("Schon angemeldet")
            time.sleep(1000)
    
        if current_time == reload_time:
            reload_page()
            click_anmeldebutton()
            if browser.find_element(By.XPATH, "/html/body/div/div/b").text == "Die Anmeldung zur Veranstaltung 4736 wurde durchgeführt.":
                print("Anmeldung erfolgreich")
                time.sleep(20000)
                break
        elif current_time > reload_time:
            while True: 
                reload_page()
                click_anmeldebutton()
                time.sleep(20000)
        


# Schließe den Browser

except:
    while True:
        print("Final Error --> Going to sleep")
        time.sleep(600000)
        browser.close()