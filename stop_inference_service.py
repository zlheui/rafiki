from rafiki.client import Client

client = Client(admin_host='localhost', admin_port=8000)
client.login(email='superadmin@rafiki', password='rafiki')
client.stop_inference_job(app='fashion_mnist_app')