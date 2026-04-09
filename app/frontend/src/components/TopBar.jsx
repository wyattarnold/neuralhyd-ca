export default function TopBar({ onInfoClick }) {
  return (
    <header className="h-12 bg-paper border-b border-gray-200 flex items-center px-4 shrink-0 z-[1100]">
      {/* DWR logo — left */}
      <img
        src="/dwr-logo.png"
        alt="California Department of Water Resources"
        className="h-8 object-contain"
      />

      {/* App title */}
      <h1 className="ml-4 text-base font-semibold text-gray-800 whitespace-nowrap select-none">
        NeuralHydrology Explorer
      </h1>
      <span className="ml-2 text-xs font-medium text-gray-400 whitespace-nowrap select-none">
        v0.1 (alpha)
      </span>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Info icon — right */}
      <button
        onClick={onInfoClick}
        className="p-1.5 rounded-full hover:bg-gray-100 transition-colors text-gray-600 hover:text-blue-600"
        title="Documentation"
        aria-label="Open documentation"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="w-5 h-5"
        >
          <circle cx="12" cy="12" r="10" />
          <path d="M12 16v-4" />
          <path d="M12 8h.01" />
        </svg>
      </button>
    </header>
  );
}
