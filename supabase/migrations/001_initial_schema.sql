-- FlowLoop Database Schema
-- Run this in Supabase SQL Editor

-- =============================================================================
-- PROFILES TABLE
-- =============================================================================
-- Extends Supabase auth.users with additional profile data

CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  display_name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Auto-create profile on user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, display_name)
  VALUES (NEW.id, NEW.raw_user_meta_data->>'display_name');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS profiles_updated_at ON public.profiles;
CREATE TRIGGER profiles_updated_at
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- =============================================================================
-- SESSIONS TABLE
-- =============================================================================
-- Training session metadata and aggregates

CREATE TABLE IF NOT EXISTS public.sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ended_at TIMESTAMPTZ,
  total_tasks INTEGER NOT NULL DEFAULT 0,
  correct_tasks INTEGER NOT NULL DEFAULT 0,
  avg_engagement REAL NOT NULL DEFAULT 0,
  avg_frustration REAL NOT NULL DEFAULT 0,
  difficulty_progression JSONB NOT NULL DEFAULT '[]'::JSONB,
  task_type TEXT NOT NULL DEFAULT 'math',
  is_adaptive BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON public.sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON public.sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_user_started ON public.sessions(user_id, started_at DESC);

-- =============================================================================
-- SESSION_EVENTS TABLE
-- =============================================================================
-- Time-series events during session (emotions, tasks, difficulty changes)

CREATE TABLE IF NOT EXISTS public.session_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
  event_type TEXT NOT NULL CHECK (event_type IN ('emotion', 'task_complete', 'difficulty_change')),
  data JSONB NOT NULL DEFAULT '{}'::JSONB,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for time-series queries
CREATE INDEX IF NOT EXISTS idx_session_events_session_id ON public.session_events(session_id);
CREATE INDEX IF NOT EXISTS idx_session_events_timestamp ON public.session_events(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_session_events_type ON public.session_events(session_id, event_type);

-- =============================================================================
-- TASK_RESULTS TABLE
-- =============================================================================
-- Individual task attempts with performance data

CREATE TABLE IF NOT EXISTS public.task_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
  task_type TEXT NOT NULL,
  difficulty INTEGER NOT NULL CHECK (difficulty >= 1 AND difficulty <= 10),
  is_correct BOOLEAN NOT NULL,
  time_spent_ms INTEGER NOT NULL,
  engagement_at_task REAL NOT NULL DEFAULT 0,
  frustration_at_task REAL NOT NULL DEFAULT 0,
  question TEXT NOT NULL,
  user_answer TEXT NOT NULL,
  correct_answer TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_task_results_session_id ON public.task_results(session_id);
CREATE INDEX IF NOT EXISTS idx_task_results_created_at ON public.task_results(session_id, created_at);

-- =============================================================================
-- ROW LEVEL SECURITY (RLS)
-- =============================================================================

-- Enable RLS on all tables
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.session_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.task_results ENABLE ROW LEVEL SECURITY;

-- Profiles: Users can only read/update their own profile
CREATE POLICY "Users can view own profile"
  ON public.profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON public.profiles FOR UPDATE
  USING (auth.uid() = id);

-- Sessions: Users can CRUD their own sessions
CREATE POLICY "Users can view own sessions"
  ON public.sessions FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can create own sessions"
  ON public.sessions FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own sessions"
  ON public.sessions FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own sessions"
  ON public.sessions FOR DELETE
  USING (auth.uid() = user_id);

-- Session Events: Users can CRUD events for their sessions
CREATE POLICY "Users can view own session events"
  ON public.session_events FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.sessions
      WHERE sessions.id = session_events.session_id
      AND sessions.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can create events for own sessions"
  ON public.session_events FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.sessions
      WHERE sessions.id = session_events.session_id
      AND sessions.user_id = auth.uid()
    )
  );

-- Task Results: Users can CRUD results for their sessions
CREATE POLICY "Users can view own task results"
  ON public.task_results FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.sessions
      WHERE sessions.id = task_results.session_id
      AND sessions.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can create results for own sessions"
  ON public.task_results FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.sessions
      WHERE sessions.id = task_results.session_id
      AND sessions.user_id = auth.uid()
    )
  );

-- =============================================================================
-- GRANTS
-- =============================================================================

GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
