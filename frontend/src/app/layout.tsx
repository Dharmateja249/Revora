import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Revora - Adaptive AI Revenue Recovery Agent",
  description: "Adaptive, policy-bounded revenue recovery agent for failed Razorpay payments.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
