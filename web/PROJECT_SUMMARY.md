# Project Summary: Legal RAG Web UI

## Overview

This directory contains a Next.js-based web interface for the Indian Legal RAG (Retrieval-Augmented Generation) system. The application provides an interactive platform for querying Indian legal documents from the ILDC dataset with real-time retrieval quality metrics.

## What Has Been Created

### Core Application Files

1. **Next.js Configuration**
   - `package.json` - Project dependencies and scripts
   - `next.config.ts` - Next.js configuration
   - `tsconfig.json` - TypeScript compiler configuration
   - `tailwind.config.ts` - Tailwind CSS theme configuration
   - `postcss.config.mjs` - PostCSS plugins configuration

2. **Application Structure**
   - `app/layout.tsx` - Root layout with metadata
   - `app/page.tsx` - Main page with chat and score display
   - `app/globals.css` - Global styles with Tailwind directives
   - `app/api/query/route.ts` - API route for backend integration (placeholder)

3. **React Components**
   - `components/ChatInterface.tsx` - Interactive chat UI with message history
   - `components/CJPEScore.tsx` - Real-time CJPE score visualization

4. **Utilities**
   - `lib/api.ts` - API client utilities for backend communication
   - `types/index.ts` - TypeScript type definitions

5. **Documentation**
   - `README.md` - Comprehensive project documentation
   - `QUICKSTART.md` - Quick start guide (5-minute setup)
   - `INTEGRATION.md` - Detailed backend integration instructions

6. **Configuration**
   - `.env.example` - Environment variable template
   - `.gitignore` - Git ignore rules for Next.js

## Key Features

### 1. Chat Interface
- Message history with timestamps
- User/assistant message differentiation
- Loading states with animations
- Auto-scroll to latest message
- Error handling
- Input validation

### 2. CJPE Score Display
- Real-time score updates (0-100%)
- Color-coded confidence levels (high/medium/low)
- Progress bar visualization
- Loading states
- Informative empty states

### 3. System Information Panel
- Model information (InLegalBERT, Mistral-7B)
- Dataset information (ILDC CJPE)
- Backend connection status
- Research disclaimer notice

### 4. Responsive Design
- Desktop and mobile optimized
- Dark mode support
- Accessible UI components
- Clean, professional styling

## Technology Stack

- **Framework:** Next.js 15 (App Router)
- **UI Library:** React 18
- **Language:** TypeScript 5
- **Styling:** Tailwind CSS 3
- **Build Tools:** PostCSS, Autoprefixer

## Current Status

### Implemented
- Complete UI/UX design
- Component architecture
- Type definitions
- API client utilities
- Responsive layout
- Dark mode support
- Mock data for testing

### Pending Integration
- Backend connection (placeholder code in place)
- Real RAG system responses
- Actual CJPE score calculation
- Case retrieval display
- Error handling for backend failures

## Getting Started

### Installation
```bash
cd web
npm install
```

### Development
```bash
npm run dev
```
Access at: http://localhost:3000

### Production Build
```bash
npm run build
npm start
```

## Integration Points

### Frontend → Backend
The frontend expects these backend endpoints:

1. **POST /query**
   - Input: `{ query: string }`
   - Output: `{ answer, cjpe_score, retrieved_cases, metadata }`

2. **GET /health**
   - Output: `{ status, version, models }`

3. **GET /metrics** (optional)
   - Output: `{ total_queries, avg_cjpe_score, ... }`

### Implementation Locations
- API route proxy: `app/api/query/route.ts` (lines 49-77)
- Chat component: `components/ChatInterface.tsx` (lines 49-77)
- API utilities: `lib/api.ts`

## File Structure

```
web/
├── app/
│   ├── api/
│   │   └── query/
│   │       └── route.ts          # Backend API proxy (PLACEHOLDER)
│   ├── layout.tsx                # Root layout
│   ├── page.tsx                  # Main page
│   └── globals.css               # Global styles
├── components/
│   ├── ChatInterface.tsx         # Chat UI
│   └── CJPEScore.tsx            # Score display
├── lib/
│   └── api.ts                    # API client utilities
├── types/
│   └── index.ts                  # TypeScript definitions
├── public/                       # Static assets
├── README.md                     # Main documentation
├── QUICKSTART.md                 # Quick start guide
├── INTEGRATION.md                # Backend integration guide
├── package.json                  # Dependencies
├── tsconfig.json                 # TypeScript config
├── tailwind.config.ts            # Tailwind config
├── next.config.ts                # Next.js config
├── postcss.config.mjs            # PostCSS config
└── .env.example                  # Environment template
```

## Next Steps for Integration

1. **Start Python Backend**
   - Implement FastAPI endpoints following `INTEGRATION.md`
   - Expose POST /query, GET /health endpoints
   - Enable CORS or use Next.js proxy

2. **Update Frontend**
   - Replace placeholder in `app/api/query/route.ts`
   - Test with real backend responses
   - Handle edge cases and errors

3. **Environment Setup**
   - Copy `.env.example` to `.env.local`
   - Set `NEXT_PUBLIC_API_URL` to backend URL

4. **Testing**
   - Verify message flow
   - Check CJPE score updates
   - Test error handling
   - Validate type safety

## Design Decisions

### Why Next.js?
- Built-in API routes for backend proxy
- Server-side rendering capabilities
- Excellent TypeScript support
- Production-ready out of the box
- Great developer experience

### Why App Router?
- Modern React patterns (Server Components)
- Better performance
- Simplified data fetching
- Future-proof architecture

### Why Tailwind CSS?
- Rapid development
- Consistent design system
- Easy dark mode implementation
- Minimal bundle size
- Highly customizable

### Component Architecture
- Separation of concerns (Chat vs Score)
- Reusable components
- Type-safe props
- Client-side interactivity where needed

## Research Context

This is a research prototype for Indian employment and workplace law. The system combines:
- **Retrieval:** InLegalBERT for semantic search
- **Generation:** Mistral-7B-Instruct for answer synthesis
- **Dataset:** ILDC CJPE subset (Indian Legal Documents Corpus)
- **Vector Store:** FAISS for similarity search

All responses should be validated by qualified legal professionals before action is taken.

## Troubleshooting

### TypeScript Errors
Normal before `npm install`. Run installation to resolve.

### Port 3000 in Use
Use alternate port: `npm run dev -- -p 3001`

### Dependencies Installation Failed
```bash
rm -rf node_modules package-lock.json
npm install
```

### Backend Not Connecting
- Verify backend is running
- Check CORS configuration
- Verify API URL in `.env.local`
- Check browser console for errors

## Performance Considerations

- Optimized bundle size with dynamic imports
- Lazy loading of components
- Efficient re-renders with React hooks
- Tailwind CSS tree-shaking
- Production build optimization

## Security Notes

- API keys should never be committed
- Use environment variables for secrets
- Implement rate limiting in production
- Add authentication if needed
- Validate and sanitize user input

## License

See root repository for licensing information.

## Support

For questions about:
- **Frontend:** See README.md and QUICKSTART.md
- **Integration:** See INTEGRATION.md
- **Backend:** See root repository documentation