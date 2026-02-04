import os
import json
import time
import fcntl
from datetime import datetime, timedelta
from pathlib import Path

class SharedRateLimiter:

    def __init__(self,
                 state_file='rate_limiter_state.json',
                 hourly_limit=3600,
                 daily_limit=50000,
                 min_delay=1.0):
        self.state_file = Path(state_file)
        self.hourly_limit = hourly_limit
        self.daily_limit = daily_limit
        self.min_delay = min_delay

        if not self.state_file.exists():
            self._write_state({
                'hourly_calls': [],
                'daily_calls': [],
                'last_call_time': None
            })

    def _read_state(self):
        try:
            with open(self.state_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    content = f.read().strip()
                    if not content:
                        return {
                            'hourly_calls': [],
                            'daily_calls': [],
                            'last_call_time': None
                        }
                    state = json.loads(content)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            state['hourly_calls'] = [datetime.fromisoformat(t) for t in state['hourly_calls']]
            state['daily_calls'] = [datetime.fromisoformat(t) for t in state['daily_calls']]
            if state['last_call_time']:
                state['last_call_time'] = datetime.fromisoformat(state['last_call_time'])

            return state
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARNING] Rate limiter state file corrupted, using default: {e}")
            return {
                'hourly_calls': [],
                'daily_calls': [],
                'last_call_time': None
            }

    def _write_state(self, state):
        state_serializable = {
            'hourly_calls': [t.isoformat() for t in state['hourly_calls']],
            'daily_calls': [t.isoformat() for t in state['daily_calls']],
            'last_call_time': state['last_call_time'].isoformat() if state['last_call_time'] else None
        }

        with open(self.state_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state_serializable, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _clean_old_entries(self, state, now):
        state['hourly_calls'] = [t for t in state['hourly_calls']
                                if now - t < timedelta(hours=1)]
        state['daily_calls'] = [t for t in state['daily_calls']
                               if now - t < timedelta(days=1)]
        return state

    def acquire(self, timeout=3600):
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                return False

            state = self._read_state()
            now = datetime.now()

            state = self._clean_old_entries(state, now)

            if len(state['hourly_calls']) >= self.hourly_limit:
                wait_time = 3600 - (now - state['hourly_calls'][0]).total_seconds()
                if wait_time > 0:
                    print(f"[{now}] Hourly limit reached, waiting {wait_time:.0f}s...")
                    time.sleep(min(wait_time, 60))
                    continue

            if len(state['daily_calls']) >= self.daily_limit:
                wait_time = 86400 - (now - state['daily_calls'][0]).total_seconds()
                if wait_time > 0:
                    print(f"[{now}] Daily limit reached, waiting {wait_time/3600:.1f}h...")
                    time.sleep(min(wait_time, 300))
                    continue

            if state['last_call_time']:
                elapsed = (now - state['last_call_time']).total_seconds()
                if elapsed < self.min_delay:
                    time.sleep(self.min_delay - elapsed)
                    continue

            now = datetime.now()
            state['hourly_calls'].append(now)
            state['daily_calls'].append(now)
            state['last_call_time'] = now

            self._write_state(state)

            return True

    def get_stats(self):
        state = self._read_state()
        now = datetime.now()
        state = self._clean_old_entries(state, now)

        return {
            'hourly_calls': len(state['hourly_calls']),
            'hourly_limit': self.hourly_limit,
            'hourly_remaining': self.hourly_limit - len(state['hourly_calls']),
            'daily_calls': len(state['daily_calls']),
            'daily_limit': self.daily_limit,
            'daily_remaining': self.daily_limit - len(state['daily_calls']),
            'last_call': state['last_call_time'].isoformat() if state['last_call_time'] else None
        }


if __name__ == '__main__':
    limiter = SharedRateLimiter(
        state_file='test_rate_limiter.json',
        hourly_limit=10,
        daily_limit=20,
        min_delay=0.5
    )

    print("Testing rate limiter...")
    for i in range(5):
        print(f"\nAttempt {i+1}:")
        if limiter.acquire():
            print(f"  Acquired at {datetime.now()}")
            stats = limiter.get_stats()
            print(f"  Stats: {stats['hourly_calls']}/{stats['hourly_limit']} hourly, "
                  f"{stats['daily_calls']}/{stats['daily_limit']} daily")
        else:
            print("  Failed to acquire")

    if os.path.exists('test_rate_limiter.json'):
        os.remove('test_rate_limiter.json')

    print("\nRate limiter test complete!")
