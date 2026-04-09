export default function InfoDrawer({ open, onClose }) {
  return (
    <>
      {/* Backdrop */}
      {open && (
        <div
          className="fixed inset-0 bg-black/30 z-[1200] transition-opacity"
          onClick={onClose}
        />
      )}

      {/* Drawer */}
      <div
        className={`fixed top-0 right-0 h-full w-[min(560px,85vw)] bg-paper shadow-xl z-[1300]
                     transform transition-transform duration-200 ease-in-out
                     ${open ? "translate-x-0" : "translate-x-full"}`}
      >
        {/* Header */}
        <div className="flex items-center justify-between h-12 px-4 border-b border-gray-200">
          <h2 className="text-sm font-semibold text-gray-700">Documentation</h2>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-gray-100 text-gray-500 hover:text-gray-800"
            aria-label="Close documentation"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2" strokeLinecap="round"
              strokeLinejoin="round" className="w-4 h-4">
              <path d="M18 6 6 18" /><path d="M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Sphinx docs iframe — only mount when open so it doesn't load eagerly */}
        {open && (
          <iframe
            src="/docs/overview.html"
            title="Documentation"
            className="w-full border-0"
            style={{ height: "calc(100% - 48px)" }}
          />
        )}
      </div>
    </>
  );
}
