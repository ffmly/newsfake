# 🚀 Deployment Guide

Comprehensive deployment instructions for the Arabic/Darija Fake News Detection System.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: Minimum 10GB free space
- **Network**: Internet connection for Haqiqa API access

### Software Dependencies
- Git
- Docker (optional but recommended)
- Python package manager (pip or conda)
- Text editor or IDE

## 🐳 Docker Deployment (Recommended)

### Quick Start with Docker

1. **Clone the Repository**
```bash
git clone <repository-url>
cd arabic-darija-fake-news-detection
```

2. **Build Docker Image**
```bash
docker build -t arabic-fake-news-detector .
```

3. **Run with Docker Compose**
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  web:
    build: .
    command: streamlit run web/app.py
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    depends_on:
      - api
    restart: unless-stopped
```

Run the services:
```bash
docker-compose up -d
```

### Production Dockerfile

```dockerfile
# Multi-stage build for optimized production image
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose ports
EXPOSE 5000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command
CMD ["python", "api/app.py"]
```

### Docker Commands

```bash
# Build image
docker build -t arabic-fake-news-detector:latest .

# Run container
docker run -d \
  --name fake-news-detector \
  -p 5000:5000 \
  -p 8501:8501 \
  -v $(pwd)/logs:/app/logs \
  -e FLASK_ENV=production \
  arabic-fake-news-detector:latest

# View logs
docker logs -f fake-news-detector

# Stop container
docker stop fake-news-detector

# Remove container
docker rm fake-news-detector
```

## ☁️ Cloud Deployment

### Heroku Deployment

1. **Install Heroku CLI**
```bash
# macOS
brew tap heroku/brew && brew install heroku

# Windows
download heroku-cli.exe

# Login
heroku login
```

2. **Create Heroku App**
```bash
heroku create your-app-name
```

3. **Deploy**
```bash
# Add Heroku specific files
echo "web: python api/app.py" > Procfile
echo "python-3.9.16" > runtime.txt

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

4. **Scale Dynos**
```bash
heroku ps:scale web=1:standard-2x
```

### AWS Elastic Beanstalk

1. **Install EB CLI**
```bash
pip install awsebcli
```

2. **Initialize Application**
```bash
eb init -p "Python 3.9 running on 64bit Amazon Linux 2"
```

3. **Create Environment**
```bash
eb create-env production
# Follow prompts to configure environment variables
```

4. **Deploy**
```bash
eb deploy production
```

### Google Cloud Platform

1. **Install gcloud CLI**
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $HOME/google-cloud-sdk/path.bash.inc
gcloud init
```

2. **Deploy to Cloud Run**
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/arabic-fake-news-detector

gcloud run deploy --image gcr.io/PROJECT-ID/arabic-fake-news-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure App Service

1. **Install Azure CLI**
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Login
az login
```

2. **Create Resource Group**
```bash
az group create --name fake-news-detector-rg --location eastus
```

3. **Deploy**
```bash
az webapp up \
  --resource-group fake-news-detector-rg \
  --name arabic-fake-news-detector \
  --runtime PYTHON:3.9 \
  --os-type Linux \
  --sku B1 \
  --location eastus
```

## 🖥️ Manual Deployment

### Local Development Setup

1. **Clone Repository**
```bash
git clone <repository-url>
cd arabic-darija-fake-news-detection
```

2. **Create Virtual Environment**
```bash
# Using venv
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

5. **Run Services**

**API Server:**
```bash
python api/app.py
```

**Web Interface:**
```bash
streamlit run web/app.py
```

### Production Setup

1. **System Preparation**
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3-dev python3-pip nginx supervisor
```

2. **Application Setup**
```bash
# Create application directory
sudo mkdir -p /opt/arabic-fake-news-detector
sudo chown $USER:$USER /opt/arabic-fake-news-detector

# Clone and setup
cd /opt/arabic-fake-news-detector
git clone <repository-url> .
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Systemd Service Files**

**API Service (`/etc/systemd/system/fake-news-api.service`):**
```ini
[Unit]
Description=Arabic Fake News Detection API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/arabic-fake-news-detector
Environment=PATH=/opt/arabic-fake-news-detector/venv/bin
ExecStart=/opt/arabic-fake-news-detector/venv/bin/python api/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Web Service (`/etc/systemd/system/fake-news-web.service`):**
```ini
[Unit]
Description=Arabic Fake News Detection Web Interface
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/arabic-fake-news-detector
Environment=PATH=/opt/arabic-fake-news-detector/venv/bin
ExecStart=/opt/arabic-fake-news-detector/venv/bin/streamlit run web/app.py --server.port=8501
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

4. **Nginx Configuration (`/etc/nginx/sites-available/fake-news-detector`):**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Streamlit proxy
    location / {
        proxy_pass http://127.0.0.1:8501/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

5. **Enable Services**
```bash
# Enable and start services
sudo systemctl enable fake-news-api fake-news-web
sudo systemctl start fake-news-api fake-news-web

# Enable nginx site
sudo ln -s /etc/nginx/sites-available/fake-news-detector /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Check status
sudo systemctl status fake-news-api fake-news-web nginx
```

## 🔧 Configuration

### Environment Variables

Create `.env` file with production settings:

```bash
# Production settings
FLASK_ENV=production
LOG_LEVEL=INFO
DEBUG=False

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=your-domain.com,localhost

# Performance
WORKERS=4
MAX_CONTENT_LENGTH=16777216

# Haqiqa API
HAQIQA_API_URL=https://walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space/api/predict
HAQIQA_API_KEY=your-api-key
REQUEST_TIMEOUT=30

# Feature weights
HAQIQA_WEIGHT=0.6
FEATURE_WEIGHT=0.4
```

### SSL/HTTPS Setup

1. **Generate SSL Certificate**
```bash
# Using Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

2. **Update Nginx Configuration**
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }
    
    # Previous location blocks...
}
```

## 📊 Monitoring and Logging

### Application Logging

Configure logging in `.env`:
```bash
LOG_LEVEL=INFO
LOG_FILE=/var/log/fake-news-detector/app.log
```

### Log Rotation

Create `/etc/logrotate.d/fake-news-detector`:
```
/var/log/fake-news-detector/*.log {
    daily
    missingok
    rotate 52
    compress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload fake-news-api fake-news-web
}
```

### Monitoring Setup

1. **Prometheus Metrics**
```python
# Add to requirements.txt
prometheus-client==0.14.1

# Add to api/app.py
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('fake_news_requests_total', 'Total requests')
request_duration = Histogram('fake_news_request_duration_seconds', 'Request duration')
```

2. **Grafana Dashboard**
- Set up Grafana for visualization
- Create dashboards for:
  - Request rate and response times
  - Error rates by endpoint
  - Risk score distributions
  - Language detection statistics

3. **Health Checks**
```bash
# Continuous health monitoring
curl -f http://localhost:5000/health

# With monitoring
while true; do
    curl -f http://localhost:5000/health || echo "Health check failed"
    sleep 30
done
```

## 🔒 Security Considerations

### API Security

1. **Rate Limiting**
```python
# Add to Flask app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)
```

2. **Input Validation**
```python
# Sanitize all inputs
import bleach
from werkzeug.utils import secure_filename

def sanitize_text(text):
    return bleach.clean(text, strip=True)
```

3. **CORS Configuration**
```python
# Restrict origins in production
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://your-domain.com"]
    }
})
```

### Environment Security

1. **File Permissions**
```bash
# Secure sensitive files
chmod 600 .env
chmod 600 logs/*.log

# Set proper ownership
sudo chown -R www-data:www-data /opt/arabic-fake-news-detector
```

2. **Firewall Configuration**
```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# iptables rules
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
```

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
```bash
# Find process using port
sudo lsof -i :5000
sudo lsof -i :8501

# Kill process
sudo kill -9 <PID>
```

2. **Memory Issues**
```bash
# Monitor memory usage
free -h
top

# Increase swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

3. **Permission Errors**
```bash
# Fix file permissions
sudo chown -R www-data:www-data /opt/arabic-fake-news-detector
sudo chmod -R 755 /opt/arabic-fake-news-detector
```

4. **API Connection Issues**
```bash
# Test Haqiqa API connectivity
curl -X POST \
  https://walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["test", "arabert"]}'
```

### Debug Mode

Enable debug logging:
```bash
# Set debug environment
export DEBUG=True
export LOG_LEVEL=DEBUG

# Run with verbose output
python api/app.py
```

### Performance Optimization

1. **Database Optimization** (if using database)
```python
# Connection pooling
from sqlalchemy import create_engine
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)
```

2. **Caching**
```python
# Add Redis caching
from flask_caching import Cache

cache = Cache(config={'CACHE_TYPE': 'redis'})

@app.route('/analyze', methods=['POST'])
@cache.cached(timeout=300)  # Cache for 5 minutes
def analyze_text():
    # Your code here
    pass
```

## 📈 Scaling

### Horizontal Scaling

1. **Load Balancer Setup**
```nginx
upstream fake_news_api {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location /api/ {
        proxy_pass http://fake_news_api;
    }
}
```

2. **Container Orchestration**
```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  api:
    build: .
    environment:
      - FLASK_ENV=production
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
```

### Auto-scaling

1. **Kubernetes Deployment**
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fake-news-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fake-news-detector
  template:
    metadata:
      labels:
        app: fake-news-detector
    spec:
      containers:
      - name: api
        image: arabic-fake-news-detector:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

2. **Horizontal Pod Autoscaler**
```yaml
# k8s-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fake-news-detector-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fake-news-detector
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 📋 Deployment Checklist

### Pre-deployment
- [ ] Environment configured correctly
- [ ] All dependencies installed
- [ ] SSL certificates obtained
- [ ] Firewall rules configured
- [ ] Monitoring setup
- [ ] Backup strategy planned
- [ ] Security review completed

### Post-deployment
- [ ] Services running correctly
- [ ] Health checks passing
- [ ] Load balancer configured
- [ ] Monitoring active
- [ ] Logs being collected
- [ ] Performance benchmarks met
- [ ] SSL certificate valid

### Rollback Plan

1. **Quick Rollback**
```bash
# Git rollback
git checkout previous-stable-tag
docker-compose down
docker-compose up -d

# Or for containers
docker stop current-container
docker run previous-image:tag
```

2. **Database Rollback** (if applicable)
```bash
# Restore database backup
psql -U username -d dbname < backup.sql

# Or for NoSQL
mongorestore --db dbname --collection collection backup.bson
```

---

## 🎯 Success Metrics

After deployment, monitor these key metrics:

### Performance Metrics
- **Response Time**: <2 seconds for 95% of requests
- **Throughput**: >100 requests/second
- **Error Rate**: <1% of total requests
- **Uptime**: >99.9%

### Business Metrics
- **Accuracy**: Maintain >94% detection accuracy
- **Language Coverage**: Support all target languages
- **User Satisfaction**: Positive feedback from users
- **Scalability**: Handle expected load

### Security Metrics
- **Vulnerability Scans**: Zero high-severity issues
- **Access Controls**: Proper authentication/authorization
- **Data Protection**: No data leaks
- **Compliance**: Meet relevant regulations

---

*For additional support or questions, refer to the main README.md file or create an issue in the repository.*