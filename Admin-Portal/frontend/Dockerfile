# Use Node.js 22 (Alpine for smaller image size)
FROM node:22-alpine3.18 AS builder

# Set working directory
WORKDIR /usr/src/app

# Install dependencies (exact versions from lock file)
COPY package.json package-lock.json ./
RUN npm install

# Copy the rest of the source code
COPY . .

# Build Next.js app
RUN npm run build

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
