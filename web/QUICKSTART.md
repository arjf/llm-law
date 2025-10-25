# Quick Start Guide

This guide will help you get the Legal RAG Web UI running in under 5 minutes.

## Prerequisites Check

Ensure you have Node.js installed:

```bash
node --version  # Should be 20.x or higher
npm --version   # Should be 10.x or higher
```

## Setup Steps

### 1. Install Dependencies

```bash
cd web
npm install
```

This will install all required packages including Next.js, React, TypeScript, and Tailwind CSS.

### 2. Start Development Server

```bash
npm run dev
```

The application will be available at [http://localhost:3000](http://localhost:3000)

### 3. Verify Installation

You should see:
- A chat interface on the left
- CJPE score display on the right
- System information panel showing "Backend Status: Not Connected"

## Testing the Interface

1. Type a legal question in the chat input (e.g., "What are grounds for termination?")
2. Press Send or hit Enter
3. You'll see a placeholder response
4. The CJPE score will update with a mock value

Note: These are mock responses until the backend is connected.

## Next Steps

### Connect to Backend

To integrate with the Python RAG system:

1. Start the Python backend (see main project README)
2. Create `.env.local` file:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```
3. Update `components/ChatInterface.tsx` to use real API calls
4. Update `app/api/query/route.ts` to forward requests to Python backend

### Build for Production

```bash
npm run build
npm start
```

Production build will be optimized and minified.

## Troubleshooting

### Port 3000 Already in Use

```bash
npm run dev -- -p 3001
```

### Dependencies Installation Fails

```bash
rm -rf node_modules package-lock.json
npm install
```

### TypeScript Errors

```bash
npm run build
```

Check for any type errors and fix them before proceeding.

## Project Structure Quick Reference

```
web/
├── app/                    # Next.js App Router
│   ├── api/               # API routes (backend proxy)
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Main page
│   └── globals.css        # Global styles
├── components/            # React components
│   ├── ChatInterface.tsx  # Chat UI
│   └── CJPEScore.tsx      # Score display
├── types/                 # TypeScript definitions
└── public/                # Static assets
```

## Common Tasks

### Add New Component

Create file in `components/YourComponent.tsx`:

```typescript
"use client";

export default function YourComponent() {
  return <div>Your content</div>;
}
```

Import and use in `app/page.tsx`.

### Modify Styling

Edit `tailwind.config.ts` for theme changes.
Edit `app/globals.css` for global styles.

### Add Environment Variable

1. Add to `.env.local`
2. Prefix with `NEXT_PUBLIC_` for client-side access
3. Restart dev server

## Support

For detailed information, see the main README.md file.

For backend integration details, refer to the root repository documentation.