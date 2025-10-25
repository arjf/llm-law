# LegalAI - Indian Legal Assistant

A production-ready web interface for AI-powered legal assistance focused on Indian employment and workplace law. Built with Next.js, TypeScript, and Tailwind CSS, featuring real-time streaming responses and document analysis capabilities.

## Features

- Real-time streaming AI responses powered by Mistral-7B
- Document upload and analysis
- CJPE score visualization for retrieval quality metrics
- Dark/Light theme toggle with persistent preferences
- Responsive design optimized for all devices
- Modern, production-grade UI with smooth animations
- Advanced document retrieval using InLegalBERT

## Technology Stack

- Next.js 15 (App Router)
- React 18
- TypeScript 5
- Tailwind CSS 3
- Streaming API support
- PostCSS with Autoprefixer

## Prerequisites

- Node.js 20.x or higher
- npm or yarn package manager

## Installation

Install dependencies:

```bash
npm install
```

## Development

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

The application will start with dark theme by default. Use the theme toggle in the navbar to switch between light and dark modes.

## Building for Production

Create an optimized production build:

```bash
npm run build
```

Start the production server:

```bash
npm start
```

## Project Structure

```
web/
├── app/
│   ├── api/
│   │   └── query/
│   │       └── route.ts          # Streaming API endpoint
│   ├── layout.tsx                # Root layout
│   ├── page.tsx                  # Main page
│   └── globals.css               # Global styles with theme
├── components/
│   ├── ChatInterface.tsx         # Chat UI with streaming
│   ├── CJPEScore.tsx            # Score visualization
│   ├── Navbar.tsx               # Navigation bar
│   └── ThemeToggle.tsx          # Theme switcher
├── lib/
│   └── api.ts                    # API client utilities
├── types/
│   └── index.ts                  # TypeScript definitions
├── public/                       # Static assets
└── [config files]               # Next.js, Tailwind, TypeScript
```

## Backend Integration

The frontend expects a Python backend with the following endpoint:

### POST /query/stream

Streaming endpoint for legal queries.

**Request:**
```json
{
  "query": "What are the grounds for termination?",
  "files": ["document1.pdf", "document2.pdf"]
}
```

**Response:** Server-Sent Events (SSE) stream

Text chunks followed by metadata:
```
Based on Indian employment law...

__METADATA__
{
  "cjpe_score": 0.85,
  "retrieved_cases": [...]
}
```

### Implementation Example

Update `app/api/query/route.ts` to connect to your Python backend:

```typescript
const backendUrl = process.env.BACKEND_API_URL || "http://localhost:8000";
const response = await fetch(`${backendUrl}/query/stream`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query, files }),
});

return new Response(response.body, {
  headers: {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
  },
});
```

## Environment Variables

Create a `.env.local` file:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Theme Customization

The application uses a custom black/orange theme defined in `tailwind.config.ts`:

- Primary background: `#1E1E1E`
- Secondary background: `#1F1F1F`
- Card background: `#2A2A2A`
- Primary accent: `#FF6B35` (orange)
- Border color: `#3A3A3A`

To customize colors, edit `tailwind.config.ts` in the `theme.extend.colors` section.

## Key Components

### ChatInterface

Handles user interaction and message display:
- Real-time streaming responses
- File upload support (PDF, DOC, DOCX, TXT)
- Message history with timestamps
- Loading states and animations
- Error handling

### CJPEScore

Displays retrieval quality metrics:
- Percentage score (0-100%)
- Color-coded confidence levels
- Animated progress bar
- Real-time updates

### Navbar

Application navigation:
- Logo and branding
- Navigation links
- Theme toggle
- Mobile-responsive menu

### ThemeToggle

Day/night mode switcher:
- Persistent theme preference
- Smooth transitions
- System preference detection

## Styling

The application uses a modern design system:

- Glass morphism effects
- Gradient accents
- Smooth transitions
- Custom scrollbars
- Focus states for accessibility
- Responsive breakpoints

## Performance

- Optimized bundle size with tree-shaking
- Code splitting and lazy loading
- Efficient re-renders with React hooks
- Production build minification
- Asset optimization

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Deployment

### Vercel (Recommended)

```bash
vercel deploy
```

### Docker

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## Troubleshooting

### Port Already in Use

```bash
npm run dev -- -p 3001
```

### Dependencies Installation Issues

```bash
rm -rf node_modules package-lock.json
npm install
```

### Theme Not Persisting

Clear localStorage and refresh:
```javascript
localStorage.clear()
location.reload()
```

### Streaming Not Working

Check that your backend returns proper SSE format with correct headers.

## Security Considerations

- Input validation on all user queries
- File upload restrictions (type and size)
- CORS configuration for API routes
- Environment variables for sensitive data
- Rate limiting (implement in API routes)

## Accessibility

- Semantic HTML structure
- ARIA labels for interactive elements
- Keyboard navigation support
- Focus indicators
- Color contrast compliance (WCAG 2.1 AA)

## License

See root repository for licensing information.

## Support

For detailed backend integration, see `INTEGRATION.md`.

For quick start instructions, see `QUICKSTART.md`.