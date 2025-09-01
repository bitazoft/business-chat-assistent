# Use the 22-alpine3.18 version of Node.js as the base image
FROM node:22-alpine3.18 

# Set the working directory inside the container
WORKDIR /usr/src/app  

# Copy project files to the container
COPY . ./  

# Install dependencies
RUN npm install  

# Expose port (use the PORT env if needed)
EXPOSE 3000  

# Start the application
CMD ["npm", "run", "dev"]
