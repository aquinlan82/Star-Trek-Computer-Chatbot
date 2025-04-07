# ---- Build Stage ---- (probably need to update alpine to a newer version)
FROM node:14-alpine AS build
WORKDIR /app

# Copy the package.json and package-lock.json
COPY package*.json ./

# Install node packages
RUN npm install
# RUN npm install @fontsource/cabin
# RUN npm install @emotion/react
# RUN npm install @emotion/styled
# RUN npm install vite-plugin-svgr

# Copy file prereqs
COPY public ./public
COPY src ./src
COPY index.html ./
COPY vite.config.js ./
COPY dist ./dist

# Build the React app
RUN npm run build

# # ---- Run Stage ----
FROM python:3.8-slim
WORKDIR /app

# Install python packages
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the React build from the previous stage
COPY --from=build /app/dist ./dist

# Copy the FastAPI app
COPY ./server.py ./
COPY ./assets ./assets

# Run the FastAPI app
CMD echo "Container starting..." && uvicorn server:app --host 0.0.0.0 --port 80