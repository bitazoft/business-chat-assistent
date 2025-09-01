# Use the 22-alpine3.18 version of Node.js as the base image
FROM node:22-alpine3.18 

# Set the working directory inside the container
WORKDIR /usr/src/app  

# Declare build arguments for secrets
ARG PORT
ARG DATABASE_URL
ARG JWT_SECRET
ARG NODE_ENV
ARG AWS_BUCKET_NAME
ARG AWS_ACCESSKEYID
ARG AWS_SECRETACCESSKEY
ARG AWS_REGION

# Set environment variables inside the container
ENV PORT=$PORT
ENV DATABASE_URL=$DATABASE_URL
ENV JWT_SECRET=$JWT_SECRET
ENV NODE_ENV=$NODE_ENV
ENV AWS_BUCKET_NAME=$AWS_BUCKET_NAME
ENV AWS_ACCESSKEYID=$AWS_ACCESSKEYID
ENV AWS_SECRETACCESSKEY=$AWS_SECRETACCESSKEY
ENV AWS_REGION=$AWS_REGION

# Copy project files to the container
COPY . ./  

# Install dependencies
RUN npm install  

# Expose port (use the PORT env if needed)
EXPOSE 7001  

# Start the application
CMD ["npm", "run", "dev"]
