{\rtf1\ansi\ansicpg1252\cocoartf2759
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red89\green156\blue62;\red23\green23\blue23;\red202\green202\blue202;
\red113\green184\blue255;\red183\green111\blue179;\red212\green212\blue212;\red194\green126\blue101;\red140\green211\blue254;
\red167\green197\blue152;\red70\green137\blue204;\red212\green214\blue154;\red67\green192\blue160;}
{\*\expandedcolortbl;;\cssrgb\c41569\c66275\c30980;\cssrgb\c11765\c11765\c11765;\cssrgb\c83137\c83137\c83137;
\cssrgb\c50980\c77647\c100000;\cssrgb\c77255\c52549\c75294;\cssrgb\c86275\c86275\c86275;\cssrgb\c80784\c56863\c47059;\cssrgb\c61176\c86275\c99608;
\cssrgb\c70980\c80784\c65882;\cssrgb\c33725\c61176\c83922;\cssrgb\c86275\c86275\c66667;\cssrgb\c30588\c78824\c69020;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 # Uses a pre-trained language model (T5) for text generation\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Implements a Flask API for serving predictions\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Includes basic error handling and logging\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Uses configuration for model parameters\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Demonstrates multi-threading for running the Flask app in Colab\cf4 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 # it's recommended to build modular code 1.Separate the code into different modules (e.g., model.py, api.py, config.py)\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 #2.Use a main.py file to tie everything together \cf4 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 # industry level development and best practices\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Modularity: Split code into separate modules (e.g., model.py, api.py, config.py) for better organization.\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Configuration Management: Use tools like python-dotenv or hydra for cleaner config handling.\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Experiment Tracking: Integrate MLflow or Weights & Biases for model versioning and tracking.\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Input/Output Handling: Add robust input validation, preprocessing, and output postprocessing.\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Caching: Use Redis or Memcached to cache frequent requests.\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Monitoring: Add comprehensive logging and monitoring with Prometheus or Grafana.\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Testing: Implement unit tests, integration tests, and CI/CD pipelines.\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Performance: Optimize inference with batching or model quantization.\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Scalability: Design for horizontal scaling (e.g., Kubernetes).\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # Documentation: Add detailed docstrings and API documentation.\cf4 \cb1 \strokec4 \
\
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 !\cf4 \strokec4 pip install transformers torch flask datasets pyyaml\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 import\cf4 \strokec4  torch\cb1 \
\cf6 \cb3 \strokec6 from\cf4 \strokec4  transformers \cf6 \strokec6 import\cf4 \strokec4  AutoTokenizer\cf7 \strokec7 ,\cf4 \strokec4  AutoModelForSeq2SeqLM\cb1 \
\cf6 \cb3 \strokec6 from\cf4 \strokec4  flask \cf6 \strokec6 import\cf4 \strokec4  Flask\cf7 \strokec7 ,\cf4 \strokec4  request\cf7 \strokec7 ,\cf4 \strokec4  jsonify\cb1 \
\cf6 \cb3 \strokec6 import\cf4 \strokec4  logging\cb1 \
\cf6 \cb3 \strokec6 import\cf4 \strokec4  os\cb1 \
\cf6 \cb3 \strokec6 import\cf4 \strokec4  yaml\cb1 \
\cf6 \cb3 \strokec6 from\cf4 \strokec4  threading \cf6 \strokec6 import\cf4 \strokec4  Thread\cb1 \
\cf6 \cb3 \strokec6 import\cf4 \strokec4  time\cb1 \
\cf6 \cb3 \strokec6 import\cf4 \strokec4  requests\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Logging setup\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 logging.basicConfig\cf7 \strokec7 (\cf4 \strokec4 level=logging.INFO\cf7 \strokec7 ,\cf4 \strokec4  format=\cf8 \strokec8 "%(asctime)s - %(levelname)s - %(message)s"\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\cb3 logger = logging.getLogger\cf7 \strokec7 (\cf9 \strokec9 __name__\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Configuration\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 config = \cf7 \strokec7 \{\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 'api_key'\cf7 \strokec7 :\cf4 \strokec4  \cf8 \strokec8 'YOUR_SIMULATED_API_KEY'\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 'model_name'\cf7 \strokec7 :\cf4 \strokec4  \cf8 \strokec8 't5-base'\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 'max_length'\cf7 \strokec7 :\cf4 \strokec4  \cf10 \strokec10 100\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 'temperature'\cf7 \strokec7 :\cf4 \strokec4  \cf10 \strokec10 0.8\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 'top_k'\cf7 \strokec7 :\cf4 \strokec4  \cf10 \strokec10 50\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 'top_p'\cf7 \strokec7 :\cf4 \strokec4  \cf10 \strokec10 0.95\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 'num_beams'\cf7 \strokec7 :\cf4 \strokec4  \cf10 \strokec10 5\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 'no_repeat_ngram_size'\cf7 \strokec7 :\cf4 \strokec4  \cf10 \strokec10 2\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 'repetition_penalty'\cf7 \strokec7 :\cf4 \strokec4  \cf10 \strokec10 1.5\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 'host'\cf7 \strokec7 :\cf4 \strokec4  \cf8 \strokec8 '0.0.0.0'\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 'port'\cf7 \strokec7 :\cf4 \strokec4  \cf10 \strokec10 5001\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf7 \cb3 \strokec7 \}\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 API_KEY = config\cf7 \strokec7 [\cf8 \strokec8 'api_key'\cf7 \strokec7 ]\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 class\cf4 \strokec4  ContentGenerator\cf7 \strokec7 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf11 \strokec11 def\cf4 \strokec4  \cf12 \strokec12 __init__\cf4 \strokec4 (\cf9 \strokec9 self\cf4 \strokec4 , \cf9 \strokec9 model_name\cf4 \strokec4 =config\cf7 \strokec7 [\cf8 \strokec8 'model_name'\cf7 \strokec7 ]\cf4 \strokec4 )\cf7 \strokec7 :\cf4 \cb1 \strokec4 \
\cb3         \cf9 \strokec9 self\cf4 \strokec4 .tokenizer = AutoTokenizer.from_pretrained\cf7 \strokec7 (\cf4 \strokec4 model_name\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\cb3         \cf9 \strokec9 self\cf4 \strokec4 .model = AutoModelForSeq2SeqLM.from_pretrained\cf7 \strokec7 (\cf4 \strokec4 model_name\cf7 \strokec7 )\cf4 \strokec4 .to\cf7 \strokec7 (\cf8 \strokec8 "cuda"\cf4 \strokec4  \cf6 \strokec6 if\cf4 \strokec4  torch.cuda.is_available\cf7 \strokec7 ()\cf4 \strokec4  \cf6 \strokec6 else\cf4 \strokec4  \cf8 \strokec8 "cpu"\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\
\cb3     \cf11 \strokec11 def\cf4 \strokec4  \cf12 \strokec12 generate_content\cf4 \strokec4 (\cf9 \strokec9 self\cf4 \strokec4 , \cf9 \strokec9 product_description\cf4 \strokec4 , \cf9 \strokec9 target_audience\cf4 \strokec4 , \cf9 \strokec9 desired_tone\cf4 \strokec4 , \cf9 \strokec9 max_length\cf4 \strokec4 =config\cf7 \strokec7 [\cf8 \strokec8 'max_length'\cf7 \strokec7 ]\cf4 \strokec4 )\cf7 \strokec7 :\cf4 \cb1 \strokec4 \
\cb3         \cf6 \strokec6 try\cf7 \strokec7 :\cf4 \cb1 \strokec4 \
\cb3             prompt = \cf11 \strokec11 f\cf8 \strokec8 "Write a \cf7 \strokec7 \{\cf4 \strokec4 desired_tone\cf7 \strokec7 \}\cf8 \strokec8  marketing copy for \cf7 \strokec7 \{\cf4 \strokec4 product_description\cf7 \strokec7 \}\cf8 \strokec8  targeting \cf7 \strokec7 \{\cf4 \strokec4 target_audience\cf7 \strokec7 \}\cf8 \strokec8 ."\cf4 \cb1 \strokec4 \
\cb3             input_ids = \cf9 \strokec9 self\cf4 \strokec4 .tokenizer\cf7 \strokec7 (\cf4 \strokec4 prompt\cf7 \strokec7 ,\cf4 \strokec4  return_tensors=\cf8 \strokec8 "pt"\cf7 \strokec7 )\cf4 \strokec4 .input_ids.to\cf7 \strokec7 (\cf9 \strokec9 self\cf4 \strokec4 .model.device\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\cb3             output = \cf9 \strokec9 self\cf4 \strokec4 .model.generate\cf7 \strokec7 (\cf4 \cb1 \strokec4 \
\cb3                 input_ids\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3                 max_length=max_length\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3                 temperature=config\cf7 \strokec7 [\cf8 \strokec8 'temperature'\cf7 \strokec7 ],\cf4 \cb1 \strokec4 \
\cb3                 top_k=config\cf7 \strokec7 [\cf8 \strokec8 'top_k'\cf7 \strokec7 ],\cf4 \cb1 \strokec4 \
\cb3                 top_p=config\cf7 \strokec7 [\cf8 \strokec8 'top_p'\cf7 \strokec7 ],\cf4 \cb1 \strokec4 \
\cb3                 num_beams=config\cf7 \strokec7 [\cf8 \strokec8 'num_beams'\cf7 \strokec7 ],\cf4 \cb1 \strokec4 \
\cb3                 no_repeat_ngram_size=config\cf7 \strokec7 [\cf8 \strokec8 'no_repeat_ngram_size'\cf7 \strokec7 ],\cf4 \cb1 \strokec4 \
\cb3                 repetition_penalty=config\cf7 \strokec7 [\cf8 \strokec8 'repetition_penalty'\cf7 \strokec7 ]\cf4 \cb1 \strokec4 \
\cb3             \cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\cb3             generated_text = \cf9 \strokec9 self\cf4 \strokec4 .tokenizer.decode\cf7 \strokec7 (\cf4 \strokec4 output\cf7 \strokec7 [\cf10 \strokec10 0\cf7 \strokec7 ],\cf4 \strokec4  skip_special_tokens=\cf11 \strokec11 True\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\cb3             \cf6 \strokec6 return\cf4 \strokec4  generated_text\cb1 \
\cb3         \cf6 \strokec6 except\cf4 \strokec4  Exception \cf6 \strokec6 as\cf4 \strokec4  e\cf7 \strokec7 :\cf4 \cb1 \strokec4 \
\cb3             logger.error\cf7 \strokec7 (\cf11 \strokec11 f\cf8 \strokec8 "Error generating content: \cf7 \strokec7 \{\cf4 \strokec4 e\cf7 \strokec7 \}\cf8 \strokec8 "\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\cb3             \cf6 \strokec6 raise\cf4 \strokec4  e\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Flask app setup\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 app = Flask\cf7 \strokec7 (\cf9 \strokec9 __name__\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\cb3 content_generator = ContentGenerator\cf7 \strokec7 ()\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 def\cf4 \strokec4  \cf12 \strokec12 authenticate\cf4 \strokec4 (\cf9 \strokec9 request\cf4 \strokec4 )\cf7 \strokec7 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     api_key = request.headers.get\cf7 \strokec7 (\cf8 \strokec8 "X-Api-Key"\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\cb3     \cf6 \strokec6 return\cf4 \strokec4  api_key == API_KEY\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf12 \cb3 \strokec12 @app.route\cf7 \strokec7 (\cf8 \strokec8 "/generate"\cf7 \strokec7 ,\cf4 \strokec4  methods=\cf7 \strokec7 [\cf8 \strokec8 "POST"\cf7 \strokec7 ])\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 def\cf4 \strokec4  \cf12 \strokec12 generate\cf4 \strokec4 ()\cf7 \strokec7 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf6 \strokec6 if\cf4 \strokec4  \cf5 \strokec5 not\cf4 \strokec4  authenticate\cf7 \strokec7 (\cf4 \strokec4 request\cf7 \strokec7 ):\cf4 \cb1 \strokec4 \
\cb3         \cf6 \strokec6 return\cf4 \strokec4  jsonify\cf7 \strokec7 (\{\cf8 \strokec8 "error"\cf7 \strokec7 :\cf4 \strokec4  \cf8 \strokec8 "Unauthorized"\cf7 \strokec7 \}),\cf4 \strokec4  \cf10 \strokec10 401\cf4 \cb1 \strokec4 \
\
\cb3     \cf6 \strokec6 try\cf7 \strokec7 :\cf4 \cb1 \strokec4 \
\cb3         data = request.get_json\cf7 \strokec7 ()\cf4 \cb1 \strokec4 \
\cb3         product_description = data.get\cf7 \strokec7 (\cf8 \strokec8 "product_description"\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\cb3         target_audience = data.get\cf7 \strokec7 (\cf8 \strokec8 "target_audience"\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\cb3         desired_tone = data.get\cf7 \strokec7 (\cf8 \strokec8 "desired_tone"\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\
\cb3         \cf6 \strokec6 if\cf4 \strokec4  \cf5 \strokec5 not\cf4 \strokec4  \cf12 \strokec12 all\cf7 \strokec7 ([\cf4 \strokec4 product_description\cf7 \strokec7 ,\cf4 \strokec4  target_audience\cf7 \strokec7 ,\cf4 \strokec4  desired_tone\cf7 \strokec7 ]):\cf4 \cb1 \strokec4 \
\cb3             \cf6 \strokec6 return\cf4 \strokec4  jsonify\cf7 \strokec7 (\{\cf8 \strokec8 "error"\cf7 \strokec7 :\cf4 \strokec4  \cf8 \strokec8 "Missing required parameters"\cf7 \strokec7 \}),\cf4 \strokec4  \cf10 \strokec10 400\cf4 \cb1 \strokec4 \
\
\cb3         content = content_generator.generate_content\cf7 \strokec7 (\cf4 \strokec4 product_description\cf7 \strokec7 ,\cf4 \strokec4  target_audience\cf7 \strokec7 ,\cf4 \strokec4  desired_tone\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\
\cb3         logger.info\cf7 \strokec7 (\cf11 \strokec11 f\cf8 \strokec8 "Generated content for product: \cf7 \strokec7 \{\cf4 \strokec4 product_description\cf7 \strokec7 \}\cf8 \strokec8 , audience: \cf7 \strokec7 \{\cf4 \strokec4 target_audience\cf7 \strokec7 \}\cf8 \strokec8 , tone: \cf7 \strokec7 \{\cf4 \strokec4 desired_tone\cf7 \strokec7 \}\cf8 \strokec8 "\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\
\cb3         \cf6 \strokec6 return\cf4 \strokec4  jsonify\cf7 \strokec7 (\{\cf8 \strokec8 "content"\cf7 \strokec7 :\cf4 \strokec4  content\cf7 \strokec7 \})\cf4 \cb1 \strokec4 \
\cb3     \cf6 \strokec6 except\cf4 \strokec4  Exception \cf6 \strokec6 as\cf4 \strokec4  e\cf7 \strokec7 :\cf4 \cb1 \strokec4 \
\cb3         logger.error\cf7 \strokec7 (\cf11 \strokec11 f\cf8 \strokec8 "Error generating content: \cf7 \strokec7 \{\cf4 \strokec4 e\cf7 \strokec7 \}\cf8 \strokec8 "\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\cb3         \cf6 \strokec6 return\cf4 \strokec4  jsonify\cf7 \strokec7 (\{\cf8 \strokec8 "error"\cf7 \strokec7 :\cf4 \strokec4  \cf13 \strokec13 str\cf7 \strokec7 (\cf4 \strokec4 e\cf7 \strokec7 )\}),\cf4 \strokec4  \cf10 \strokec10 500\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Run the app in a separate thread\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf11 \cb3 \strokec11 def\cf4 \strokec4  \cf12 \strokec12 run_app\cf4 \strokec4 ()\cf7 \strokec7 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     app.run\cf7 \strokec7 (\cf4 \strokec4 debug=\cf11 \strokec11 False\cf7 \strokec7 ,\cf4 \strokec4  host=config\cf7 \strokec7 [\cf8 \strokec8 'host'\cf7 \strokec7 ],\cf4 \strokec4  port=config\cf7 \strokec7 [\cf8 \strokec8 'port'\cf7 \strokec7 ])\cf4 \cb1 \strokec4 \
\
\cb3 thread = Thread\cf7 \strokec7 (\cf4 \strokec4 target=run_app\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\cb3 thread.start\cf7 \strokec7 ()\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Wait for the app to start\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 time.sleep\cf7 \strokec7 (\cf10 \strokec10 5\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Test the API\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 headers = \cf7 \strokec7 \{\cf8 \strokec8 "X-Api-Key"\cf7 \strokec7 :\cf4 \strokec4  API_KEY\cf7 \strokec7 \}\cf4 \cb1 \strokec4 \
\cb3 data = \cf7 \strokec7 \{\cf8 \strokec8 "product_description"\cf7 \strokec7 :\cf4 \strokec4  \cf8 \strokec8 "a new AI-powered chatbot"\cf7 \strokec7 ,\cf4 \strokec4  \cf8 \strokec8 "target_audience"\cf7 \strokec7 :\cf4 \strokec4  \cf8 \strokec8 "businesses"\cf7 \strokec7 ,\cf4 \strokec4  \cf8 \strokec8 "desired_tone"\cf7 \strokec7 :\cf4 \strokec4  \cf8 \strokec8 "informative"\cf7 \strokec7 \}\cf4 \cb1 \strokec4 \
\
\cb3 response = requests.post\cf7 \strokec7 (\cf11 \strokec11 f\cf8 \strokec8 "http://\cf7 \strokec7 \{\cf4 \strokec4 config\cf7 \strokec7 [\cf8 \strokec8 'host'\cf7 \strokec7 ]\}\cf8 \strokec8 :\cf7 \strokec7 \{\cf4 \strokec4 config\cf7 \strokec7 [\cf8 \strokec8 'port'\cf7 \strokec7 ]\}\cf8 \strokec8 /generate"\cf7 \strokec7 ,\cf4 \strokec4  headers=headers\cf7 \strokec7 ,\cf4 \strokec4  json=data\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 if\cf4 \strokec4  response.status_code == \cf10 \strokec10 200\cf7 \strokec7 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf12 \strokec12 print\cf7 \strokec7 (\cf8 \strokec8 "Generated content:"\cf7 \strokec7 ,\cf4 \strokec4  response.json\cf7 \strokec7 ()[\cf8 \strokec8 "content"\cf7 \strokec7 ])\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 else\cf7 \strokec7 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     \cf12 \strokec12 print\cf7 \strokec7 (\cf8 \strokec8 "Error:"\cf7 \strokec7 ,\cf4 \strokec4  response.json\cf7 \strokec7 ())\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Additional product examples to test\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 product_examples = \cf7 \strokec7 [\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 "a sustainable clothing line for eco-conscious consumers"\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 "a new AI-powered fitness tracker"\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 "an online course for learning data science"\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 "a subscription box for organic pet food"\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 "a productivity app for managing tasks and projects"\cf7 \strokec7 ,\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf7 \cb3 \strokec7 ]\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 for\cf4 \strokec4  product_description \cf5 \strokec5 in\cf4 \strokec4  product_examples\cf7 \strokec7 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     data = \cf7 \strokec7 \{\cf8 \strokec8 "product_description"\cf7 \strokec7 :\cf4 \strokec4  product_description\cf7 \strokec7 ,\cf4 \strokec4  \cf8 \strokec8 "target_audience"\cf7 \strokec7 :\cf4 \strokec4  \cf8 \strokec8 "young adults"\cf7 \strokec7 ,\cf4 \strokec4  \cf8 \strokec8 "desired_tone"\cf7 \strokec7 :\cf4 \strokec4  \cf8 \strokec8 "persuasive"\cf7 \strokec7 \}\cf4 \cb1 \strokec4 \
\cb3     response = requests.post\cf7 \strokec7 (\cf11 \strokec11 f\cf8 \strokec8 "http://\cf7 \strokec7 \{\cf4 \strokec4 config\cf7 \strokec7 [\cf8 \strokec8 'host'\cf7 \strokec7 ]\}\cf8 \strokec8 :\cf7 \strokec7 \{\cf4 \strokec4 config\cf7 \strokec7 [\cf8 \strokec8 'port'\cf7 \strokec7 ]\}\cf8 \strokec8 /generate"\cf7 \strokec7 ,\cf4 \strokec4  headers=headers\cf7 \strokec7 ,\cf4 \strokec4  json=data\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\cb3     \cf12 \strokec12 print\cf7 \strokec7 (\cf11 \strokec11 f\cf8 \strokec8 "\\nProduct: \cf7 \strokec7 \{\cf4 \strokec4 product_description\cf7 \strokec7 \}\cf8 \strokec8 "\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\cb3     \cf12 \strokec12 print\cf7 \strokec7 (\cf8 \strokec8 "Generated content:"\cf7 \strokec7 ,\cf4 \strokec4  response.json\cf7 \strokec7 ()\cf4 \strokec4 .get\cf7 \strokec7 (\cf8 \strokec8 "content"\cf7 \strokec7 ,\cf4 \strokec4  \cf8 \strokec8 "Error: "\cf4 \strokec4  + \cf13 \strokec13 str\cf7 \strokec7 (\cf4 \strokec4 response.json\cf7 \strokec7 ())))\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Stop the Flask app\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 import\cf4 \strokec4  os\cb1 \
\cf6 \cb3 \strokec6 import\cf4 \strokec4  signal\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 os.kill\cf7 \strokec7 (\cf4 \strokec4 os.getpid\cf7 \strokec7 (),\cf4 \strokec4  signal.SIGINT\cf7 \strokec7 )\cf4 \cb1 \strokec4 \
\
}