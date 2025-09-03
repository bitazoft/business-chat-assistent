# Use Node.js 22 (Alpine for smaller image size)
FROM node:22-alpine3.18 AS builder

# Set working directory
WORKDIR /usr/src/app

# Accept build argument
# ARG NEXT_PUBLIC_API_BASE_URL

# Debug: Print what we received
# RUN echo "=== BUILD DEBUG ==="
# RUN echo "Received build arg: ${NEXT_PUBLIC_API_BASE_URL}"

# Install dependencies first
COPY package.json package-lock.json ./
RUN npm install

# Copy the rest of the source code
COPY . .

# Create .env.production file with the build arg
# RUN echo "NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}" > .env.production

# Verify the .env file was created correctly
# RUN echo "Created .env.production file:"
# RUN cat .env.production
# RUN echo "=================="

# Build Next.js app (it will read from .env.production)
RUN npm run build

# Verify the environment variable was embedded in the build
# RUN echo "Checking if variable was embedded in build:"
# RUN find /usr/src/app/.next -name "*.js" -exec grep -l "${NEXT_PUBLIC_API_BASE_URL}" {} \; 2>/dev/null | head -3 || echo "Not found in JS files"

# ---------------------
# Production Stage
# ---------------------
FROM node:22-alpine3.18 AS runner

WORKDIR /usr/src/app

# Only copy built assets + required files
COPY --from=builder /usr/src/app/package.json ./
COPY --from=builder /usr/src/app/package-lock.json ./
COPY --from=builder /usr/src/app/.next ./.next
COPY --from=builder /usr/src/app/public ./public
COPY --from=builder /usr/src/app/node_modules ./node_modules

# Expose port
EXPOSE 3000

# Run optimized Next.js app
CMD ["npm", "start"]