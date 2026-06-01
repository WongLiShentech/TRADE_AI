import { NavLink, Outlet } from 'react-router-dom'

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/signals', label: 'Signals' },
  { to: '/portfolio', label: 'Portfolio' },
  { to: '/backtester', label: 'Backtester' },
]

export default function Layout() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <nav className="border-b border-gray-800 px-6 py-3 flex gap-6">
        {navItems.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              isActive ? 'text-white font-semibold' : 'text-gray-400 hover:text-gray-200'
            }
          >
            {label}
          </NavLink>
        ))}
      </nav>
      <main className="p-6">
        <Outlet />
      </main>
    </div>
  )
}
