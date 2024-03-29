import ast
import copy
import glob
import io
import logging
import os
import pandas as pd
import smtplib
import sys
from collections import Counter
from datetime import datetime as dt
from datetime import timedelta
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
# from pyLDAvis import gensim
# from gensim.corpora import Dictionary
# from gensim.models import LdaModel, CoherenceModel
# import pyLDAvis
from io import BytesIO
# from nltk import word_tokenize, WordNetLemmatizer
# from nltk.corpus import stopwords
from wordcloud import WordCloud

# nltk.download('punkt')
# nltk.download('stopwords')
today_date = dt.today().strftime('%m-%d-%Y')
quickclean = ['how','you','will','say','said','are','has','cnet','new','says','u','bloomberg','best', 'source', 'year', 'people', 'story', 'stock']

def analyze_stories(types, bucket_name):
    result = {}
    for typ in types:
        print('Reading files for type:'+typ)
        data, files = read_files(bucket_name, typ)
        # data, files = read_local_files(typ) # local
        if data is not None and not data.empty:
            try:
                wc, top, lda, size = topic_checks(data, 'PText')  # local
            except:
                print('Error analysing type: ',typ, str(sys.exc_info()))
                continue
            result[typ]={}
            result[typ]['data'] = data
            result[typ]['files'] = files
            result[typ]['wc'] = wc
            result[typ]['top'] = top
            result[typ]['size'] = size
            result[typ]['lda'] = lda if lda is not None else ''
    return result


def read_local_files(type_):
    current_date = dt.utcnow()
    one_month_ago = current_date - timedelta(days=30)
    relevant_files = []
    file_pattern = os.path.join(type_, '*.csv')
    files = glob.glob(file_pattern)
    data = {type_: None}
    dfs = []
    for file_path in files:
        file_name = os.path.basename(file_path)
        file_date_str = file_name.split('.')[0]
        file_date = dt.strptime(file_date_str, '%Y%m%d')
        if one_month_ago <= file_date <= current_date:
            relevant_files.append(file_name)
            dfs.append(pd.read_csv(file_path))
    df = pd.concat(dfs, axis=0) if len(dfs) else None
    if any(['Processed' in x for x in df.columns]):
        df['PText'] = df['PText'].fillna(df['Processed_Text'])
    return df, relevant_files


def df_to_gcp_csv(df, dest_bucket_name, dest_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(dest_bucket_name)
    blob = bucket.blob(dest_file_name)
    blob.upload_from_string(df.to_csv(), 'text/csv')
    print(f'DataFrame uploaded to bucket {dest_bucket_name}, file name: {dest_file_name}')


def gcp_csv_to_df(bucket_name, source_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))
    print(f'Pulled down file from bucket {bucket_name}, file name: {source_file_name}')
    return df


from google.cloud import storage
period_str = ''
def read_files(bucket_name, type_):
    client = storage.Client()
    current_date = dt.utcnow()
    one_month_ago = current_date - timedelta(days=30)
    blobs = client.list_blobs(bucket_name, prefix=f'{type_}/')
    relevant_files = []
    dfs = []
    global period_str
    period_str = one_month_ago.strftime("%b %d") + " to " + current_date.strftime("%b %d")
    for blob in blobs:
        logging.info(blob.name)
        file_name = blob.name.split('/')[-1]
        file_date_str = file_name.split('.')[0]
        try:
            file_date = dt.strptime(file_date_str, '%Y%m%d')
            if one_month_ago <= file_date <= current_date:
                logging.info(file_name+'::Relevant')
                relevant_files.append(file_name)
                file_data = blob.download_as_bytes()
                df_ = pd.read_csv(io.BytesIO(file_data))
                dfs.append(df_)
        except:
            print('File read error: ', file_date_str, type_,str(sys.exc_info()))
            continue
            # logging.error('File read error: ' +file_date_str + type_ + '\n' +str(sys.exc_info()))
    df = pd.concat(dfs, axis=0) if len(dfs) else None
    logging.info("Size:: " + str(df.shape)+' :: '+type_)
    if any(['Processed' in x for x in df.columns]):
        df['PText'] = df['PText'].fillna(df['Processed_Text'])

    return df, relevant_files


def create_email_body(result):
    print('Building report')
    # Create a MIME multipart message
    msg = MIMEMultipart()
    global period_str
    html_content = f'''<h1>Weekly Topicverse</h1>
                        <h2>{today_date}</h2>
                        <h3>Context Period: {period_str}</h3>'''
    msg.attach(MIMEText(html_content, 'html'))
    for typ, data in result.items():
        if 'wc' in data and data['wc'] is not None:
            siz = str(data['size'])
            wordcloud_img = MIMEImage(data['wc'])
            wordcloud_img.add_header('Content-ID', f'<{typ}_wordcloud>')
            wordcloud_img.add_header('Content-Disposition', 'inline', filename=f'{typ}_wordcloud.png')
            msg.attach(wordcloud_img)

            html_content = f'''
            <h2>{typ}</h2>
            <h2>Articles: {siz}</h2>
            <img src="cid:{typ}_wordcloud">
            '''
            msg.attach(MIMEText(html_content, 'html'))

        if 'top' in data and data['top']!={}:
            top20 = ",".join([k for k in data['top']]) if 'top' in data else ""
            top_20_words = MIMEText(f"<h3>Top 20 words: </h3><span>{top20}</span><br><br>", 'html')
            msg.attach(top_20_words)

        # lda_html = MIMEText(data['lda'], 'html')
        # msg.attach(lda_html)

        msg.attach(MIMEText("\n____________\n\n\n", 'plain'))

    print('Email message created.')
    return msg


def email_weekly_report(result):
    if result=={} or len(result.keys())==0:
        return 0
    sender_email = 'subhayuchakr@gmail.com'
    recipient_email = os.environ.get('EMAIL_TO')
    ps_ = os.environ.get('EMAIL_PASSWORD')

    if not (sender_email and recipient_email and ps_):
        raise ValueError("Email credentials are not configured.")

    try:
        message = create_email_body(result)
        print('Sending email...')
        message['From'] = sender_email
        message['To'] = recipient_email
        message['Subject'] = f'Topicverse Snapshot: {today_date}'

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, ps_)
        server.sendmail(sender_email, recipient_email, message.as_string())
        server.quit()

    except Exception as e:
        print('Email sending failed: ', str(e))
        return 0

    return 1




def email_out(result):
    try:
        projectid = 'topicverse'
        sender_address = f"summary@[{projectid}].appspotmail.com"
        ok = make_mail_report(result)
        if ok:
            message = mail.EmailMessage(
                sender=sender_address,
                subject="Test Email")

            message.to = "Subh <subhayuchakr@gmail.com>"
            message.body = """
            Test email.
            """
            message.send()
    except:
        print('Email failed: ', str(sys.exc_info()))
        return 'Email failed: '+str(sys.exc_info())
    print('Email sent')
    return 'Email sent'



def topic_checks(data, field):
    wc = generate_wordcloud(data, field)
    top_20_terms = find_top20words(data, field)
    # lda = do_lda_html(df)
    lda = None
    size = data.shape[0]
    print(str(size))
    return wc, top_20_terms, lda, size


def generate_wordcloud(data, field):
    df = copy.deepcopy(data)
    try:
        df[field] = df[field].apply(ast.literal_eval)
    except:
        logging.error('No valid clean column for wordcloud.', str(df.columns))
        return None
    text = ' '.join([term for sublist in df[field].tolist() for term in sublist if term not in quickclean])
    wordcloud = WordCloud(width=1200, height=900, background_color='white').generate(text)
    img_bytes_io = BytesIO()
    wordcloud.to_image().save(img_bytes_io, format='PNG')
    img_bytes = img_bytes_io.getvalue()
    return img_bytes

def find_top20words(data, field):
    df = copy.deepcopy(data)
    try:
        df[field] = df[field].apply(ast.literal_eval)
    except:
        logging.error('No valid clean column for top 20.', str(df.columns))
        return {}
    all_terms = [term for sublist in df[field].tolist() for term in sublist if term not in quickclean]
    term_counts = Counter(all_terms)
    top_20_terms = dict(sorted(term_counts.items(), key=lambda item: item[1], reverse=True)[:20])
    return top_20_terms



# def do_lda_html(data, field):
#     field = 'Processed_Text'
#     processed_titles = data[field].apply(eval)
#     dictionary = Dictionary(processed_titles)
#     corpus = [dictionary.doc2bow(title) for title in processed_titles]
#     coherence_values = []
#     model_list = []
#     for num_topics in range(1, round(len(processed_titles)/5)):
#         lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
#         model_list.append(lda_model)
#         coherencemodel = CoherenceModel(model=lda_model, texts=data[field].apply(eval).to_list(), dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())

#     optimal_num_topics = coherence_values.index(max(coherence_values)) + 1
#     optimal_lda_model = LdaModel(corpus, num_topics=optimal_num_topics, id2word=dictionary)

#     print(f"Optimal Number of Topics: {optimal_num_topics}")
#     for topic_num in range(optimal_num_topics):
#         print(f"Topic {topic_num + 1}: {optimal_lda_model.print_topic(topic_num)}")

#     if optimal_num_topics > 1:
#         prepared_data = pyLDAvis.gensim.prepare(optimal_lda_model, corpus, dictionary)
#         html_string = pyLDAvis.prepared_data_to_html(prepared_data)
#         html_path = Path("output/lda_viz.html")
#         pyLDAvis.save_html(prepared_data, str(html_path))
#         return html_string
#     return None


# def do_dmm_analysis(dictionary, texts):
#     group_topics = 10
#     gsdmm = MovieGroupProcess(K=group_topics, alpha=0.1, beta=0.3, n_iters=group_topics)
#     y = gsdmm.fit(texts, len(dictionary))

#     doc_count = np.array(gsdmm.cluster_doc_count)
#     print('Number of documents per topic :', doc_count)

#     # Topics sorted by the number of document they are allocated to
#     top_index = doc_count.argsort()[-group_topics:][::-1]
#     print('Most important clusters (by number of docs inside):', top_index)

#     # define function to get top words per topic
#     def top_words(cluster_word_distribution, top_cluster, values):
#         for cluster in top_cluster:
#             sort_dicts = sorted(cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
#             print("\nCluster %s : %s" % (cluster, sort_dicts))

#     # get top words in topics
#     top_words(gsdmm.cluster_word_distribution, top_index, 20)

#     cluster_word_distribution = gsdmm.cluster_word_distribution

#     topic_num = 0
#     # Select topic you want to output as dictionary (using topic_number)
#     topic_dict = sorted(cluster_word_distribution[topic_num].items(), key=lambda k: k[1], reverse=True)  # [:values]

#     # Generate a word cloud image
#     wordcloud = WordCloud(background_color='#fcf2ed',
#                           width=1000,
#                           height=600,
#                           colormap='flag').generate_from_frequencies(topic_dict)

#     # Print to screen
#     fig, ax = plt.subplots(figsize=[20, 10])
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off");
#     # Save to disk
#     plt.savefig('dmm_summary_cloud.png')

# def do_topicwizard_analysis(dictionary, texts):
#     min_topics = 1

#     vectorizer = CountVectorizer(min_df=min_topics, max_df=5)

#     # Creating a Dirichlet Multinomial Mixture Model with 30 components
#     dmm = DMM(n_components=5, n_iterations=100, alpha=0.1, beta=0.1)

#     # Creating topic pipeline
#     pipeline = Pipeline([
#         ("vectorizer", vectorizer),
#         ("dmm", dmm),
#     ])
#     full_string = texts[0]
#     pipeline.fit(full_string)
#     topicwizard.visualize(pipeline=pipeline, corpus=full_string)

# def do_lda_analysis(df, corpus, dictionary, texts):
#     coherence_values = []
#     model_list = []
#     for num_topics in range(2, 4):
#         lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
#         model_list.append(lda_model)
#         coherencemodel = CoherenceModel(model=lda_model, texts=df['Processed_Text'], dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())
#     end_time = time.time()
#     elapsed_time = end_time - start_time

#     print(f"LDA Time: {elapsed_time} seconds")

#     optimal_num_topics = coherence_values.index(max(coherence_values)) + 2  # Adding 2 because we started the loop from 2
#     optimal_lda_model = models.LdaModel(corpus, num_topics=optimal_num_topics, id2word=dictionary)

#     print(f"Optimal Number of Topics: {optimal_num_topics}")
#     for topic_num in range(optimal_num_topics):
#         print(f"Topic {topic_num + 1}: {optimal_lda_model.print_topic(topic_num)}")

#     # topic_assignments = [max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus]
#     # # Topic Summarization
#     # topic_summaries = [lda_model.print_topic(topic_num) for topic_num in range(lda_model.num_topics)]
#     #
#     # # Quality Evaluation
#     # coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
#     # coherence_score = coherence_model.get_coherence()
#     # perplexity_score = lda_model.log_perplexity(corpus)

#     prepared_data = pyLDAvis.gensim.prepare(optimal_lda_model, corpus, dictionary)
#     # pyLDAvis.display(prepared_data)
#     # pyLDAvis.save_html(prepared_data, image_path+'topic_cluster.html')
#     html_content = pyLDAvis.prepared_data_to_html(prepared_data)
#     report_collection['data']['pyldavis_html'] = html_content

