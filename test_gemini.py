from google import genai

client = genai.Client(
    api_key="AIzaSyB85PouUBhoqMB5Y-nv6mY6G0YBFV0XnJg"
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="你好"
)

print(response.text)