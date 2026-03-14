import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const location = useLocation();

  return (
    <header className="navbar">
      <div className="navbarInner">
        <Link to="/" className="brand">
          <div className="brandMark">FS</div>
          <div>
            <div className="brandTitle">ForenSight</div>
            <div className="brandSub">Digital Image Forensics</div>
          </div>
        </Link>

        <nav className="navLinks">
          <Link
            to="/"
            className={location.pathname === "/" ? "navLink active" : "navLink"}
          >
            Home
          </Link>
          <Link
            to="/analyze"
            className={location.pathname === "/analyze" ? "navLink active" : "navLink"}
          >
            Analyze
          </Link>
        </nav>
      </div>
    </header>
  );
}