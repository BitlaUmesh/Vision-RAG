import NextAuth from "next-auth"
import CredentialsProvider from "next-auth/providers/credentials"

const handler = NextAuth({
  providers: [
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        username: { label: "Username", type: "text", placeholder: "admin" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials, req) {
        // Simple hardcoded auth for Phase 3.
        // In a real app, you would check a database here.
        if (credentials?.username === "admin" && credentials?.password === "password") {
          return { id: "1", name: "Admin", email: "admin@visionrag.local" }
        }
        return null
      }
    })
  ],
  session: {
    strategy: "jwt",
  },
  pages: {
    // Optionally redirect to a custom login page, or keep NextAuth default
    // signIn: '/login',
  },
  secret: process.env.NEXTAUTH_SECRET || "vision-rag-super-secret-key-123",
})

export { handler as GET, handler as POST }
