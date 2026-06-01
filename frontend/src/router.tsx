import { createBrowserRouter } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Signals from './pages/Signals'
import Portfolio from './pages/Portfolio'
import Backtester from './pages/Backtester'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'signals', element: <Signals /> },
      { path: 'portfolio', element: <Portfolio /> },
      { path: 'backtester', element: <Backtester /> },
    ],
  },
])
