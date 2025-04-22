import axios from 'axios';
import { useState } from 'react';

function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const login = async () => {
    try {
      const res = await axios.post('http://<your-ip>:5000/login', {
        username, password
      });
      alert(res.data.message);
    } catch {
      alert('Login failed');
    }
  };

  const register = async () => {
    try {
      const res = await axios.post('http://<your-ip>:5000/register', {
        username, password
      });
      alert(res.data.message);
    } catch {
      alert('Registration failed');
    }
  };

  return (
    <div>
      <h2>Login/Register</h2>
      <input type="text" placeholder="Username" onChange={(e) => setUsername(e.target.value)} />
      <input type="password" placeholder="Password" onChange={(e) => setPassword(e.target.value)} />
      <button onClick={login}>Login</button>
      <button onClick={register}>Register</button>
    </div>
  );
}

export default Login;
