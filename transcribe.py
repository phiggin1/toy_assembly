import roslibpy

#rework to use zmq

def handler(request, response):
    print('Setting speed to {}'.format(request['data'][0]))

    response['transcription'] = "response"

    return True

client = roslibpy.Ros(host='localhost', port=7777)

print(client.is_connected)

service = roslibpy.Service(client, '/get_transciption', 'toy_assembly/Transcription')
service.advertise(handler)
print('Service advertised.')

client.run_forever()
client.terminate()