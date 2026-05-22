# LineLens AI Dashboard

A real-time, aesthetically pleasing React dashboard built with Vite that polls your Supabase database every minute for updates from the LineLens AI pipeline.

## Setup Instructions

### 1. Supabase Configuration

This application connects directly to a Supabase database. You'll need to create a table exactly as described below.

1. Go to your Supabase project dashboard.
2. Open the **SQL Editor**.
3. Run the following query to create the table:

```sql
CREATE TABLE linelens_reports (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamp with time zone default timezone('utc'::text, now()),
  report_data jsonb
);
```

### 2. Environment Variables

Create a `.env` file in the root of the `dashboard` folder (where `package.json` is located) with your Supabase keys:

```
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

### 3. Run the Dashboard

Make sure to install dependencies first:
```bash
npm install
```

Then start the development server:
```bash
npm run dev
```

The dashboard will open automatically and poll the `linelens_reports` table every 60 seconds, fetching the most recent entry.

## Updating Python Script Keys

You also need to set the same environment variables on your machine where the Python script `run_all.py` runs, or you can add them to a `.env` file in the `FINAL` directory.

The Python uploader (`upload_to_supabase.py`) will read `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` to upload data every 24 hours or when the video ends.
