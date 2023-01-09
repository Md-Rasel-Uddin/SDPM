# Load Database Pkg
import sqlite3
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()


# Fxn
def create_page_visited_table():
	c.execute('CREATE TABLE IF NOT EXISTS pageTrackTable2(pagename TEXT,timeOfvisit TIMESTAMP)')

def add_page_visited_details(pagename,timeOfvisit):
	c.execute('INSERT INTO pageTrackTable2(pagename,timeOfvisit) VALUES(?,?)',(pagename,timeOfvisit))
	conn.commit()

def view_all_page_visited_details():
	c.execute('SELECT * FROM pageTrackTable2')
	data = c.fetchall()
	return data


# Fxn To Track Input & Prediction
def create_emotionclf_table3():
	c.execute('CREATE TABLE IF NOT EXISTS emotionclfTable3(rawtext TEXT,Predicted,Max_score,prediction TEXT,probability NUMBER,timeOfvisit TIMESTAMP)')

def add_prediction_details2(rawtext,ll,sc,prediction,probability,timeOfvisit):
	c.execute('INSERT INTO emotionclfTable3(rawtext,predicted,max_score, prediction,probability,timeOfvisit) VALUES(?,?,?,?,?,?)',(rawtext,ll,sc,prediction,probability,timeOfvisit))
	conn.commit()

def view_all_prediction_details():
	c.execute('SELECT * FROM emotionclfTable3')
	data = c.fetchall()
	return data