import sys
import time
from datetime import datetime

from flask import Flask, request, jsonify, json
import ops

from google.appengine.api import mail
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# app = create_app()
app = Flask(__name__)
# appf.wsgi_app = google.appengine.api.wrap_wsgi_app(appf.wsgi_app, use_deferred=True)
# app.mount("/v1", WSGIMiddleware(appf))

bucket_name = 'a-storyverse'
func_start_time = time.time()
today_date = datetime.today().strftime('%Y%m%d')
json_file_path='sources.json'
with open(json_file_path, 'r') as json_file:
    sources = json.load(json_file)
    print('~~~Sources loaded~~~')
# with open(json_file_path, 'r') as json_file:
#     exc_map = json.load(json_file)
#     print('~~~Exclusions loaded~~~')


@app.route('/analyzestories', methods=['POST'])
def analyze_stories():
    request_data = request.get_json()
    story_type = request_data.get('type').lower()
    # daterange = request_data.get('daterange').lower() if 'daterange' in request_data else None
    # story_type='all'
    types = list(sources.keys()) if 'all' in story_type else story_type
    print("Starting weekly story analyze...", types)
    start_time = time.time()
    try:
        result = ops.analyze_stories(types, bucket_name)
        files = {typ: result[typ]['files'] for typ in types if typ in result}
        ops.email_out(result)
    except:
        print(str(sys.exc_info()))
        return jsonify({'API': "Topicverse", 'call': "analyzestories:" + str(types), "status": 'Failure'})
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_elapsed_time = end_time - func_start_time
    print(f"Weekly Story Analyze Time: {elapsed_time} seconds")
    print(f"Run Time: {total_elapsed_time} seconds")
    print(f"Story Analysis ended for type: ", str(types))
    return jsonify({'API':"Topicverse", 'call': "analyzestories: "+str(types), "files": files, "status": 'Complete'})



@app.route('/emailtest', methods=['POST','GET'])
def email_test():
    print('Processing email test request...')
    try:
        sender_email = 'subhayuchakr@gmail.com'
        recipient_email = 'subhayuchakr@gmail.com'
        password = os.environ.get('EMAIL_PASSWORD')

        if not (sender_email and recipient_email and password):
            raise ValueError("Email credentials are not configured.")

        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = recipient_email
        message['Subject'] = "Test Email"

        body = "Test email."
        message.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, recipient_email, message.as_string())
        server.quit()

    except Exception as e:
        print('Email failed: ', str(e))
        return jsonify({'API':"Topicverse", 'call': "emailtest:", "status": 'Failure'})

    print('Email sent')
    return jsonify({'API':"Topicverse", 'call': "emailtest:", "status": 'Complete'})


@app.route('/', methods=['POST', 'GET'])
def test():
    print('Reached API test')
    return jsonify({"API":"Topicverse", "Version": '1.0'})


@app.route('/param', methods=['POST'])
def params():
    print('Reached API test params', str(request.args))
    return jsonify({"API":"Topicverse", "Version": '1.0'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
