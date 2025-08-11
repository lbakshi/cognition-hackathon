# Voxe Frontend

Beautiful ML experiment dashboard built with Next.js 15, TypeScript, and Tailwind CSS.

## 🚀 Railway Deployment

This project is **Railway-ready**! Deploy with one click:

### Method 1: Direct Repository Deploy
1. Connect your GitHub repo to Railway
2. Railway auto-detects Next.js and deploys automatically
3. No additional configuration needed!

### Method 2: CLI Deploy
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Deploy from this directory
railway deploy
```

### Method 3: GitHub Integration
1. Push this code to your GitHub repository
2. Connect repository to Railway
3. Railway will auto-deploy on every push to main branch

## 🔧 Environment Variables (Optional)

If you need to connect to external APIs, set these in Railway dashboard:

- `NEXT_PUBLIC_API_URL` - Your backend API URL
- `NODE_ENV` - Set to `production` for Railway

## 📦 Build Configuration

Railway will automatically:
- Install dependencies: `npm install`
- Build the project: `npm run build` 
- Start the server: `npm start`
- Serve on Railway's provided port

## 🎯 Integration Ready

The app is designed to easily integrate with your Railway-deployed backend:

```typescript
// Replace demo data with your Railway API
const response = await fetch('https://your-backend.railway.app/api/experiment', {
  method: 'POST',
  body: JSON.stringify({ prompt })
})
```

## 🛠 Local Development

```bash
npm install
npm run dev
```

Open http://localhost:3000 to see the dashboard.

## 📊 Features

- ✅ Research prompt input with examples
- ✅ Real-time experiment tracking
- ✅ Interactive metric charts (accuracy, precision, recall, f1-score)
- ✅ Model comparison dashboards  
- ✅ Beautiful Vercel-style UI with glass morphism
- ✅ Fully responsive design
- ✅ TypeScript for type safety
- ✅ Railway deployment ready

Built for the Cognition Hackathon 🚀
