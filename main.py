from dotenv import load_dotenv
import os
load_dotenv(encoding='utf-8-sig',verbose=True)
def main():
    print("Hello from langchain-course!")
    print(os.environ.get('GEMINI_API_KEY'))

if __name__ == "__main__":
    main()
