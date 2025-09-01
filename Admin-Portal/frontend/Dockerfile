# Use the 22-alpine3.18 version of Node.js as the base image
FROM node:22-alpine3.18 

# Set the working directory inside the container
WORKDIR /usr/src/app  

# Copy only package.json & lock first for caching
COPY package.json package-lock.json ./

RUN npm install --legacy-peer-deps --omit=optional

RUN npm ci --legacy-peer-deps --omit=optional

COPY . .

RUN npm run build

# Expose port (use the PORT env if needed)
EXPOSE 3000  

# Start the application
CMD ["npm", "run", "dev"]
