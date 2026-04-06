"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/utils/supabase/client";
import { CheckCircle2 } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const supabase = createClient();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    confirmPassword: "",
    token: "",
  });
  const [status, setStatus] = useState<"idle" | "otp_sent">("idle");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSendOTP = async (e: React.MouseEvent) => {
    e.preventDefault();
    setError(null);

    if (!formData.email || !formData.password || !formData.confirmPassword) {
      setError("Please fill out email, password, and confirm password.");
      return;
    }
    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    const { error: signUpError } = await supabase.auth.signUp({
      email: formData.email,
      password: formData.password,
    });

    if (signUpError) {
      setError(signUpError.message);
      setSuccess(null);
    } else {
      setStatus("otp_sent");
      setError(null);
      setSuccess("OTP sent successfully! Please check your inbox.");
      setTimeout(() => setSuccess(null), 5000);
    }
    setLoading(false);
  };

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (!formData.token) {
      setError("Please enter the OTP.");
      return;
    }

    setLoading(true);
    const { error: verifyError } = await supabase.auth.verifyOtp({
      email: formData.email,
      token: formData.token,
      type: "signup",
    });

    if (verifyError) {
      setError(verifyError.message);
      setLoading(false);
    } else {
      router.push("/");
      router.refresh();
    }
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (!formData.email || !formData.password) {
      setError("Please enter both email and password.");
      return;
    }

    setLoading(true);
    const { error: loginError } = await supabase.auth.signInWithPassword({
      email: formData.email,
      password: formData.password,
    });

    if (loginError) {
      setError(loginError.message);
      setLoading(false);
    } else {
      router.push("/");
      router.refresh();
    }
  };

  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-zinc-950 p-4 relative">
      {success && (
        <div className="fixed top-6 right-6 z-50 rounded-xl bg-emerald-950/80 p-4 text-sm font-medium text-emerald-400 border border-emerald-900/50 shadow-2xl animate-in fade-in slide-in-from-top-4 flex items-center gap-2 backdrop-blur-md">
          <CheckCircle2 className="w-5 h-5 text-emerald-400" />
          {success}
        </div>
      )}

      <div className="w-full max-w-md p-8 bg-zinc-900 border border-zinc-800 rounded-2xl shadow-2xl transition-all duration-300">
        <h2 className="text-3xl justify-center flex font-bold tracking-tight text-white mb-8">
          Vision RAG
        </h2>

        {error && (
          <div className="mb-6 rounded-lg bg-red-950/30 p-4 text-sm text-red-400 border border-red-900/50 animate-in fade-in slide-in-from-top-2">
            {error}
          </div>
        )}

        {mode === "signup" ? (
          <form onSubmit={handleSignUp} className="flex flex-col gap-5" autoComplete="off">
            <div className="space-y-1 mb-2">
              <h3 className="text-zinc-200 font-semibold text-lg">Create your account</h3>
              <p className="text-zinc-500 text-sm">Sign up with your email and password.</p>
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium text-zinc-400 ml-1" htmlFor="signup-email">
                Login mail id
              </label>
              <input
                id="signup-email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                className="rounded-xl border border-zinc-700 bg-zinc-800/50 px-4 py-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-zinc-600 focus:border-transparent placeholder:text-zinc-600 transition-all"
                placeholder="email@example.com"
                required
                autoComplete="off"
                disabled={status === "otp_sent"}
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium text-zinc-400 ml-1" htmlFor="signup-password">
                password
              </label>
              <input
                id="signup-password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                className="rounded-xl border border-zinc-700 bg-zinc-800/50 px-4 py-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-zinc-600 focus:border-transparent placeholder:text-zinc-600 transition-all"
                placeholder="••••••••"
                required
                minLength={6}
                autoComplete="new-password"
                disabled={status === "otp_sent"}
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium text-zinc-400 ml-1" htmlFor="signup-confirm-password">
                re-enter password
              </label>
              <input
                id="signup-confirm-password"
                name="confirmPassword"
                type="password"
                value={formData.confirmPassword}
                onChange={handleChange}
                className="rounded-xl border border-zinc-700 bg-zinc-800/50 px-4 py-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-zinc-600 focus:border-transparent placeholder:text-zinc-600 transition-all"
                placeholder="••••••••"
                required
                minLength={6}
                autoComplete="new-password"
                disabled={status === "otp_sent"}
              />
            </div>

            <div className="flex gap-3 items-end">
              <div className="flex-1 flex flex-col gap-1.5">
                <input
                  id="signup-token"
                  name="token"
                  type="text"
                  value={formData.token}
                  onChange={handleChange}
                  className="rounded-xl border border-zinc-700 bg-zinc-800/50 px-4 py-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-zinc-600 focus:border-transparent placeholder:text-zinc-600 transition-all"
                  placeholder="OTP"
                  maxLength={10}
                  disabled={status !== "otp_sent"}
                />
              </div>
              <button
                type="button"
                onClick={handleSendOTP}
                disabled={loading || status === 'otp_sent'}
                className="rounded-xl bg-zinc-700 px-4 py-3 text-sm font-bold text-white hover:bg-zinc-600 transition-all active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
              >
                {status === "otp_sent" ? "OTP Sent" : "Send OTP"}
              </button>
            </div>

            <button
              type="submit"
              disabled={loading || status !== "otp_sent"}
              className="mt-4 w-full rounded-xl bg-zinc-100 py-3 text-sm font-bold text-zinc-900 hover:bg-white transition-all active:scale-[0.98] shadow-lg shadow-white/5 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Sign Up
            </button>

            <div className="mt-4 text-center">
              <p className="text-sm text-zinc-500">
                Already have an account?{" "}
                <button
                  type="button"
                  onClick={() => {
                    setMode("login");
                    setError(null);
                    setSuccess(null);
                    setStatus("idle");
                  }}
                  className="text-zinc-200 font-semibold hover:underline"
                >
                  Sign in
                </button>
              </p>
            </div>
          </form>
        ) : (
          <form onSubmit={handleLogin} className="flex flex-col gap-5" autoComplete="off">
            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium text-zinc-400 ml-1" htmlFor="login-email">
                Login Mail
              </label>
              <input
                id="login-email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                className="rounded-xl border border-zinc-700 bg-zinc-800/50 px-4 py-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-zinc-600 focus:border-transparent placeholder:text-zinc-600 transition-all"
                placeholder="email@example.com"
                required
                autoComplete="off"
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium text-zinc-400 ml-1" htmlFor="login-password">
                Password
              </label>
              <input
                id="login-password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                className="rounded-xl border border-zinc-700 bg-zinc-800/50 px-4 py-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-zinc-600 focus:border-transparent placeholder:text-zinc-600 transition-all"
                placeholder="••••••••"
                required
                autoComplete="new-password"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="mt-4 w-full rounded-xl bg-zinc-100 py-3 text-sm font-bold text-zinc-900 hover:bg-white transition-all active:scale-[0.98] shadow-lg shadow-white/5 disabled:opacity-50"
            >
              Login
            </button>

            <div className="mt-4 text-center">
              <p className="text-sm text-zinc-500">
                Don&apos;t have an account?{" "}
                <button
                  type="button"
                  onClick={() => {
                    setMode("signup");
                    setError(null);
                    setSuccess(null);
                    setStatus("idle");
                  }}
                  className="text-zinc-200 font-semibold hover:underline"
                >
                  Create one
                </button>
              </p>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
