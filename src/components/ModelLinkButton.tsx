import { Link } from "react-router-dom";

type Props = {
  to: string;
  label: string;
};

export default function ModelLinkButton({ to, label }: Props) {
  return (
    <Link to={to} className="model-link-btn">
      {label}
    </Link>
  );
}
