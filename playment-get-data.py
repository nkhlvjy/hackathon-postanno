import psycopg2
import requests
import time, uuid


def call_api(job_id, url):
    # print(job_id)
    headers = {'content-type': 'application/json','Authorization':'Bearer eyJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE2MjYzNjE4NzgsImlkIjoiM2NlZWRjYjUtNWU5Ny00ODBiLTkxNzQtYTZkMTJhMzcxMWYxIiwicm9sZXMiOlsiYWRtaW4iXSwicHdkX3VwZGF0ZWRfYXQiOjE1NTAyMzkwNzk3Nzd9.MyIGzYVjJ-1Yp-T1VjxYGvYQCDwv6ZM9PMsjz_JdhJs'}
    response = requests.get(url, headers= headers)
    if response.status_code == 200:
        print("sucessfully fetched the data")
        return response
    else:
        print(
            f"Hello person, there's a {response.status_code} error with your request")


if __name__ == "__main__":
    start = time.time()
    # batch_id = uuid.UUID('762d368b-b220-49f3-80ae-dfe4e6264f62')
    engine = psycopg2.connect(
        database="playment_api_production",
        user="ulquiorra865",
        password="a8P$n!perPl@yCoD#",
        host="db-replica.playment.io",
        port='5432'
    )
    cursor = engine.cursor()

    image_from_build_query = '''select id,
           build::json -> 'image_url'          as image_url,
           build::json -> 'image_url_original' as image_url_original
    from feed_line
    where batch_id = '762d368b-b220-49f3-80ae-dfe4e6264f62';'''
    cursor.execute(image_from_build_query)
    print('Queried the table successfully: obtained image_url, image_url_original and job_id for all jobs in the batch')

    for i in cursor.fetchall():
        # print('https://luigi.playment.io/api/v1/attachments?url=' + i[1])
        response = call_api(i[0], 'https://luigi.playment.io/api/v1/attachments?url=' + i[1])
        print(response)
        with open("/Users/playment/Documents/playment-data/original/"+i[0]+".jpg", 'wb') as f:
            f.write(response.content)

    submission_result_url_from_submissions_query = '''select id, submission_result_url
        from submissions
        where batch_id = '762d368b-b220-49f3-80ae-dfe4e6264f62';'''
    cursor.execute(submission_result_url_from_submissions_query)
    print('Queried the table successfully: obtained job_id, submission_result_url for all jobs in the batch')

    for i in cursor.fetchall():
        response = call_api(i[0], 'https://luigi.playment.io/api/v1/attachments?url=' + i[1])
        print(response.json()['url'])
        response = call_api(i[0], 'https://luigi.playment.io/api/v1/attachments?url=' + response.json()['url'])
        print(response)
        with open("/Users/playment/Documents/playment-data/annotated/"+i[0]+".png", 'wb') as f:
            f.write(response.content)

    end = time.time()
    print(f"time taken: {end - start}")
    engine.commit()
    engine.close()