# AGCO Sales AI Interface

Contains react-js front end code and fastapi back end code for the AGCO Sales AI application

Navigate to interface folder:
```bash
cd sales_ai
cd interface
```


To install & build (Windows):
Torch assumes you have CUDA 11.8 installed

```bash
npm install
npm install @fontsource/cabin
npm install @emotion/react
npm install @emotion/styled
npm install vite-plugin-svgr
npm install react-markdown
npm run build
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118 
python -m pip install -r requirements.txt
```

To run:
```bash
uvicorn server:app
```

To docker-run:
```bash
docker build -t fendt_product_expert .
docker run -p 8000:80 --env-file ".env" -i -t fendt_product_expert
```