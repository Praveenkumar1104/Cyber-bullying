import React from 'react'
import NavBar from './components/NavBar'
import { Outlet } from 'react-router-dom'

const Main = () => {
  return (
    <>
    <div>
    <div className="fixed top-0 h-20 w-full bg-white z-50">
    <NavBar/>
    </div>
    <div className='absolute top-0 z-10'>
    <Outlet/>
    </div>
    </div>
    </>
  )
}

export default Main