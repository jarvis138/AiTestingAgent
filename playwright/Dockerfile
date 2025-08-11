FROM node:18-slim
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npx playwright install --with-deps
CMD ["npm", "test"]
