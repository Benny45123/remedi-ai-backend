-- seed.sql — demo data for MediAssist.
-- Run this in the Supabase SQL editor before every demo (or via psql).
-- Wipes existing doctors/slots first so re-running it is always safe and
-- gives you a clean, fully-open slot board.

BEGIN;

-- Clean slate. Order matters: appointments/patients reference slots/doctors.
DELETE FROM appointments;
DELETE FROM patients;
DELETE FROM doctor_slots;
DELETE FROM doctors;

-- ---------- Doctors ----------

INSERT INTO doctors (id, name, specialization) VALUES
    ('11111111-1111-1111-1111-111111111111', 'Dr. Mehta',  'Dermatologist'),
    ('22222222-2222-2222-2222-222222222222', 'Dr. Rao',    'General Physician'),
    ('33333333-3333-3333-3333-333333333333', 'Dr. Iyer',   'Pediatrician');

-- ---------- Slots ----------
-- ~6 open slots per doctor, spread across "tomorrow" and "the day after",
-- at typical clinic hours. Using now()::date + interval keeps this script
-- correct no matter what day you run it on before a demo.

INSERT INTO doctor_slots (doctor_id, slot_start, status) VALUES
    -- Dr. Mehta (Dermatologist)
    ('11111111-1111-1111-1111-111111111111', (now()::date + interval '1 day' + interval '10 hours'), 'open'),
    ('11111111-1111-1111-1111-111111111111', (now()::date + interval '1 day' + interval '11 hours'), 'open'),
    ('11111111-1111-1111-1111-111111111111', (now()::date + interval '1 day' + interval '16 hours'), 'open'),
    ('11111111-1111-1111-1111-111111111111', (now()::date + interval '2 days' + interval '10 hours'), 'open'),
    ('11111111-1111-1111-1111-111111111111', (now()::date + interval '2 days' + interval '11 hours'), 'open'),
    ('11111111-1111-1111-1111-111111111111', (now()::date + interval '2 days' + interval '17 hours'), 'open'),

    -- Dr. Rao (General Physician)
    ('22222222-2222-2222-2222-222222222222', (now()::date + interval '1 day' + interval '9 hours'),  'open'),
    ('22222222-2222-2222-2222-222222222222', (now()::date + interval '1 day' + interval '12 hours'), 'open'),
    ('22222222-2222-2222-2222-222222222222', (now()::date + interval '1 day' + interval '15 hours'), 'open'),
    ('22222222-2222-2222-2222-222222222222', (now()::date + interval '2 days' + interval '9 hours'),  'open'),
    ('22222222-2222-2222-2222-222222222222', (now()::date + interval '2 days' + interval '13 hours'), 'open'),
    ('22222222-2222-2222-2222-222222222222', (now()::date + interval '2 days' + interval '16 hours'), 'open'),

    -- Dr. Iyer (Pediatrician)
    ('33333333-3333-3333-3333-333333333333', (now()::date + interval '1 day' + interval '10 hours 30 minutes'), 'open'),
    ('33333333-3333-3333-3333-333333333333', (now()::date + interval '1 day' + interval '13 hours'),            'open'),
    ('33333333-3333-3333-3333-333333333333', (now()::date + interval '1 day' + interval '17 hours'),            'open'),
    ('33333333-3333-3333-3333-333333333333', (now()::date + interval '2 days' + interval '10 hours 30 minutes'), 'open'),
    ('33333333-3333-3333-3333-333333333333', (now()::date + interval '2 days' + interval '12 hours'),            'open'),
    ('33333333-3333-3333-3333-333333333333', (now()::date + interval '2 days' + interval '16 hours'),            'open');

COMMIT;