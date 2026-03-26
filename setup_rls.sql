-- Désactiver RLS pour permettre les opérations via la clé anon
-- (à sécuriser plus tard avec des policies si besoin)
alter table documents enable row level security;
alter table chunks enable row level security;

-- Policies pour permettre toutes les opérations
create policy "Allow all on documents" on documents for all using (true) with check (true);
create policy "Allow all on chunks" on chunks for all using (true) with check (true);
